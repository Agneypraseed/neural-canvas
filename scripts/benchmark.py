"""Run a reproducible, machine-readable neural-style-transfer benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import numpy
import PIL
import torch
import torchvision
from PIL import Image
from torchvision.models import VGG19_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neural_style_transfer.config import StyleTransferConfig  # noqa: E402
from neural_style_transfer.engine import resolve_device, run_style_transfer  # noqa: E402
from neural_style_transfer.image_io import load_image, save_image  # noqa: E402
from neural_style_transfer.model import VGG19FeatureExtractor  # noqa: E402

SCHEMA_NAME = "neural-canvas-benchmark"
SCHEMA_VERSION = 2
SUPPORTED_OUTPUT_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
FINAL_LOSS_SEMANTICS = (
    "The total is the weighted objective; content, style, and total_variation are raw "
    "unweighted components. The engine re-evaluated these values after the optimizer update "
    "and clamp for the reported step, so they describe the returned result tensor and callback "
    "image. The encoded output file may include normal 8-bit quantization."
)


def _existing_input(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input image does not exist or is not a file: {path}")
    return path


def _output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        suffixes = ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        raise argparse.ArgumentTypeError(f"output must use one of these extensions: {suffixes}")
    if path.exists() and path.is_dir():
        raise argparse.ArgumentTypeError(f"output path is a directory: {path}")
    return path


def _image_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be an integer") from exc
    if parsed < 32:
        raise argparse.ArgumentTypeError("size must be at least 32 pixels")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without running a benchmark."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the real pretrained VGG-19 style-transfer pipeline and emit JSON "
            "metadata to stdout. The first run may download the official model weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--content", required=True, type=_existing_input, help="content image")
    parser.add_argument("--style", required=True, type=_existing_input, help="style image")
    parser.add_argument(
        "--size",
        required=True,
        type=_image_size,
        help="maximum long-edge size used while loading each input",
    )
    parser.add_argument(
        "--steps", required=True, type=_positive_int, help="number of Adam optimization steps"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="execution device",
    )
    parser.add_argument("--output", required=True, type=_output_path, help="result image path")
    parser.add_argument("--seed", default=42, type=int, help="PyTorch random seed")
    parser.add_argument(
        "--progress-interval",
        default=25,
        type=_positive_int,
        help="engine loss-snapshot interval",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vgg_weight_cache_path() -> Path:
    filename = Path(urlparse(VGG19_Weights.DEFAULT.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _timezone_payload(timestamp: datetime) -> dict[str, str | None]:
    raw_offset = timestamp.strftime("%z")
    utc_offset = f"{raw_offset[:3]}:{raw_offset[3:]}" if raw_offset else None
    return {"name": timestamp.tzname(), "utc_offset": utc_offset}


def _environment_payload() -> dict[str, object]:
    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "description": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "Pillow": str(PIL.__version__),
            "numpy": str(numpy.__version__),
        },
        "cpu": {
            "processor": processor,
            "logical_count": os.cpu_count(),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
    }


def _device_payload(requested: str, resolved: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {"requested": requested, "resolved": str(resolved)}
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "name": properties.name,
                "index": index,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
                "cuda_runtime": torch.version.cuda,
            }
        )
    elif resolved.type == "mps":
        payload["name"] = "Apple Metal Performance Shaders"
    else:
        payload["name"] = platform.processor() or platform.machine() or None
    return payload


def _tensor_dimensions(tensor: torch.Tensor) -> dict[str, int]:
    return {"width": int(tensor.shape[-1]), "height": int(tensor.shape[-2])}


def _validate_preprocessed_image(tensor: torch.Tensor, label: str) -> None:
    minimum_edge = min(int(tensor.shape[-2]), int(tensor.shape[-1]))
    if minimum_edge < 16:
        raise ValueError(
            f"{label} is too small after preprocessing; each edge must be at least 16 pixels"
        )


def _paths_collide(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Execute one benchmark and return its JSON-serializable report."""

    content_path = args.content.resolve()
    style_path = args.style.resolve()
    output_path = args.output.resolve()
    if _paths_collide(output_path, content_path) or _paths_collide(output_path, style_path):
        raise ValueError("output must not overwrite a content or style input")

    timestamp = datetime.now().astimezone()
    resolved_device = resolve_device(args.device)
    environment = _environment_payload()
    device = _device_payload(args.device, resolved_device)
    config = StyleTransferConfig(
        image_size=args.size,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )

    total_started = perf_counter()
    weight_cache_path = _vgg_weight_cache_path()
    weight_cached_before = weight_cache_path.is_file()
    content_sha256 = _sha256(content_path)
    style_sha256 = _sha256(style_path)
    content = load_image(content_path, config.image_size)
    style = load_image(style_path, config.image_size)
    _validate_preprocessed_image(content, "content image")
    _validate_preprocessed_image(style, "style image")

    requested_layers = {config.content_layer, *(name for name, _ in config.style_layers)}
    extractor = VGG19FeatureExtractor(requested_layers).to(resolved_device).eval()
    _synchronize(resolved_device)
    load_finished = perf_counter()

    render_started = perf_counter()
    result = run_style_transfer(content, style, config, feature_extractor=extractor)
    _synchronize(resolved_device)
    render_finished = perf_counter()
    if not result.history:
        raise RuntimeError("style-transfer engine returned no loss snapshots")

    saved_path = save_image(result.image, output_path).resolve()
    with Image.open(saved_path) as saved_image:
        output_dimensions = {"width": saved_image.width, "height": saved_image.height}
        output_format = saved_image.format
    output_sha256 = _sha256(saved_path)
    total_finished = perf_counter()

    final_snapshot = result.history[-1]
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "timezone": _timezone_payload(timestamp),
        "config": asdict(config),
        "device": {**device, "engine_reported": result.device},
        "environment": environment,
        "weights": {
            "identifier": str(VGG19_Weights.DEFAULT),
            "cached_before": weight_cached_before,
            "cached_after": weight_cache_path.is_file(),
        },
        "inputs": {
            "content": {
                "path": str(content_path),
                "sha256": content_sha256,
                "preprocessed_dimensions": _tensor_dimensions(content),
            },
            "style": {
                "path": str(style_path),
                "sha256": style_sha256,
                "preprocessed_dimensions": _tensor_dimensions(style),
            },
        },
        "timings_seconds": {
            "load_and_preprocess": load_finished - total_started,
            "render": render_finished - render_started,
            "output_encode_and_hash": total_finished - render_finished,
            "total": total_finished - total_started,
        },
        "timing_semantics": {
            "load_and_preprocess": (
                "input hashing, decoding/resizing, VGG-19 construction or weight download, "
                "and transfer to the selected device"
            ),
            "render": "run_style_transfer, including optimization and device synchronization",
            "total": "load/preprocess, render, output encoding, and output hashing",
        },
        "output": {
            "path": str(saved_path),
            "sha256": output_sha256,
            "dimensions": output_dimensions,
            "format": output_format,
        },
        "final_loss_snapshot": {
            "semantics": FINAL_LOSS_SEMANTICS,
            "step": final_snapshot.step,
            "total": final_snapshot.total,
            "content": final_snapshot.content,
            "style": final_snapshot.style,
            "total_variation": final_snapshot.total_variation,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"benchmark failed: {exc}\n")
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

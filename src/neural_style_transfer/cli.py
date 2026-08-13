"""Command-line interface for Neural Canvas."""

import argparse
import math
import os
import sys
from pathlib import Path

from neural_style_transfer.config import StyleTransferConfig
from neural_style_transfer.engine import LossSnapshot, run_style_transfer
from neural_style_transfer.image_io import (
    load_image,
    load_pil_image,
    preserve_content_colors,
    save_image,
    tensor_to_pil,
)


def _image_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 32:
        raise argparse.ArgumentTypeError("must be at least 32")
    return parsed


def _positive_steps(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nst",
        description="Combine the structure of one image with the visual style of another.",
    )
    parser.add_argument("content", type=Path, help="path to the content image")
    parser.add_argument("style", type=Path, help="path to the style image")
    parser.add_argument(
        "--output", type=Path, default=Path("output/stylized.png"), help="output image path"
    )
    parser.add_argument("--size", type=_image_size, default=512, help="maximum content-image edge")
    parser.add_argument("--steps", type=_positive_steps, default=300, help="optimization steps")
    parser.add_argument("--learning-rate", type=_positive_float, default=0.02)
    parser.add_argument("--content-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--style-weight", type=_nonnegative_float, default=100_000.0)
    parser.add_argument("--tv-weight", type=_nonnegative_float, default=0.0001)
    parser.add_argument("--initialization", choices=("content", "noise"), default="content")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preserve-color",
        action="store_true",
        help="preserve the content image's color channels",
    )
    return parser


def _print_progress(snapshot: LossSnapshot, _image: object) -> None:
    print(
        f"step {snapshot.step:>4} | total {snapshot.total:.5f} | "
        f"content {snapshot.content:.5f} | style {snapshot.style:.5f}"
    )


def _canonical_path(path: Path) -> str:
    """Normalize a path for collision checks, including existing symlinks."""

    return os.path.normcase(str(path.resolve(strict=False)))


def _paths_collide(first: Path, second: Path) -> bool:
    if _canonical_path(first) == _canonical_path(second):
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def _validate_output_path(output: Path, content: Path, style: Path) -> None:
    if _paths_collide(output, content) or _paths_collide(output, style):
        raise ValueError("output path must differ from the content and style input paths")


def _run(args: argparse.Namespace) -> None:
    _validate_output_path(args.output, args.content, args.style)
    config = StyleTransferConfig(
        image_size=args.size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        total_variation_weight=args.tv_weight,
        initialization=args.initialization,
        device=args.device,
        seed=args.seed,
        progress_interval=max(1, args.steps // 10),
    )

    content = load_image(args.content, config.image_size)
    style = load_image(args.style, config.image_size)
    result = run_style_transfer(content, style, config, callback=_print_progress)

    if args.preserve_color:
        generated = tensor_to_pil(result.image)
        original_content = load_pil_image(args.content)
        generated = preserve_content_colors(generated, original_content)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generated.save(args.output)
    else:
        save_image(result.image, args.output)

    print(f"saved {args.output} using {result.device}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _run(args)
    except (MemoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

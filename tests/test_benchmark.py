from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest
import torch
from PIL import Image

from neural_style_transfer.config import StyleTransferConfig
from neural_style_transfer.engine import LossSnapshot, StyleTransferResult
from scripts import benchmark


class FakeFeatureExtractor:
    requested_layers: set[str] | None = None
    device: torch.device | None = None

    def __init__(self, requested_layers: set[str]) -> None:
        type(self).requested_layers = requested_layers

    def to(self, device: torch.device) -> FakeFeatureExtractor:
        type(self).device = device
        return self

    def eval(self) -> FakeFeatureExtractor:
        return self


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def test_benchmark_emits_measured_json_without_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content_path = tmp_path / "content.png"
    style_path = tmp_path / "style.png"
    output_path = tmp_path / "nested" / "result.png"
    _write_image(content_path, (24, 20), (12, 34, 56))
    _write_image(style_path, (20, 24), (78, 90, 123))

    def fake_run_style_transfer(
        content: torch.Tensor,
        style: torch.Tensor,
        config: StyleTransferConfig,
        *,
        feature_extractor: FakeFeatureExtractor,
    ) -> StyleTransferResult:
        assert content.shape == (1, 3, 20, 24)
        assert style.shape == (1, 3, 24, 20)
        assert feature_extractor is not None
        assert config.steps == 2
        return StyleTransferResult(
            image=torch.full_like(content, 0.5),
            history=(
                LossSnapshot(
                    step=2,
                    total=4.0,
                    content=1.0,
                    style=2.0,
                    total_variation=3.0,
                ),
            ),
            device="cpu",
        )

    monkeypatch.setattr(benchmark, "VGG19FeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(benchmark, "run_style_transfer", fake_run_style_transfer)
    monkeypatch.setattr(
        benchmark, "_vgg_weight_cache_path", lambda: tmp_path / "uncached-vgg19.pth"
    )

    exit_code = benchmark.main(
        [
            "--content",
            str(content_path),
            "--style",
            str(style_path),
            "--size",
            "32",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "neural-canvas-benchmark"
    assert report["schema_version"] == 2
    assert datetime.fromisoformat(report["timestamp"]).tzinfo is not None
    assert report["timezone"]["utc_offset"] is not None
    assert report["config"]["image_size"] == 32
    assert report["config"]["steps"] == 2
    assert report["config"]["seed"] == 42
    assert report["device"]["requested"] == "cpu"
    assert report["device"]["resolved"] == "cpu"
    assert report["device"]["engine_reported"] == "cpu"
    assert set(report["environment"]["versions"]) == {
        "Pillow",
        "numpy",
        "torch",
        "torchvision",
    }
    assert report["environment"]["cpu"]["logical_count"] is not None
    assert report["environment"]["cpu"]["torch_intraop_threads"] >= 1
    assert report["weights"]["cached_before"] is False
    assert report["weights"]["cached_after"] is False
    assert FakeFeatureExtractor.device == torch.device("cpu")
    assert FakeFeatureExtractor.requested_layers == {
        "relu1_1",
        "relu2_1",
        "relu3_1",
        "relu4_1",
        "relu4_2",
        "relu5_1",
    }
    assert all(value >= 0 for value in report["timings_seconds"].values())
    assert report["timings_seconds"]["total"] >= report["timings_seconds"]["render"]
    assert report["output"]["path"] == str(output_path.resolve())
    assert report["output"]["dimensions"] == {"width": 24, "height": 20}
    assert report["output"]["format"] == "PNG"
    assert report["output"]["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert report["final_loss_snapshot"]["step"] == 2
    assert report["final_loss_snapshot"]["total"] == 4.0
    assert "after the optimizer update and clamp" in report["final_loss_snapshot"]["semantics"]


def test_benchmark_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        benchmark.main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--content" in help_text
    assert "--style" in help_text
    assert "--size" in help_text
    assert "--steps" in help_text
    assert "--device" in help_text
    assert "--output" in help_text


@pytest.mark.parametrize(
    ("changed", "value", "message"),
    [
        ("--content", "missing.png", "does not exist"),
        ("--size", "31", "at least 32"),
        ("--steps", "0", "at least 1"),
        ("--progress-interval", "0", "at least 1"),
        ("--output", "result.bmp", "extensions"),
    ],
)
def test_benchmark_argument_validation(
    changed: str,
    value: str,
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    content = tmp_path / "content.png"
    style = tmp_path / "style.png"
    _write_image(content, (32, 24), (12, 34, 56))
    _write_image(style, (24, 32), (78, 90, 123))
    arguments = {
        "--content": str(content),
        "--style": str(style),
        "--size": "32",
        "--steps": "1",
        "--progress-interval": "1",
        "--output": str(tmp_path / "result.png"),
    }
    arguments[changed] = str(tmp_path / value) if changed in {"--content", "--output"} else value
    argv = [item for pair in arguments.items() for item in pair]

    with pytest.raises(SystemExit) as invalid_exit:
        benchmark.main(argv)
    assert invalid_exit.value.code == 2
    assert message in capsys.readouterr().err


def test_benchmark_rejects_hard_link_output_before_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content = tmp_path / "content.png"
    style = tmp_path / "style.png"
    output = tmp_path / "content-alias.png"
    _write_image(content, (32, 24), (12, 34, 56))
    _write_image(style, (24, 32), (78, 90, 123))
    output.hardlink_to(content)
    original = content.read_bytes()

    def unexpected_transfer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("benchmark must reject an input alias before rendering")

    monkeypatch.setattr(benchmark, "run_style_transfer", unexpected_transfer)

    with pytest.raises(SystemExit) as error:
        benchmark.main(
            [
                "--content",
                str(content),
                "--style",
                str(style),
                "--size",
                "32",
                "--steps",
                "1",
                "--output",
                str(output),
            ]
        )

    assert error.value.code == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert content.read_bytes() == original

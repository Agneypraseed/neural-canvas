from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from neural_style_transfer import cli


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    content = tmp_path / "content.png"
    style = tmp_path / "style.png"
    Image.new("RGB", (40, 32), (80, 120, 160)).save(content)
    Image.new("RGB", (32, 40), (160, 120, 80)).save(style)
    return content, style


def _mock_transfer(
    content: torch.Tensor, _style: torch.Tensor, _config: object, **_kwargs: object
) -> SimpleNamespace:
    return SimpleNamespace(image=content, device="cpu")


def test_cli_runs_offline_with_mocked_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content, style = _write_inputs(tmp_path)
    output = tmp_path / "nested" / "result.png"
    monkeypatch.setattr(cli, "run_style_transfer", _mock_transfer)

    exit_code = cli.main(
        [
            str(content),
            str(style),
            "--output",
            str(output),
            "--size",
            "32",
            "--steps",
            "1",
        ]
    )

    assert exit_code == 0
    assert output.is_file()
    with Image.open(output) as result:
        assert result.size == (32, 26)
    assert f"saved {output} using cpu" in capsys.readouterr().out


@pytest.mark.parametrize("input_name", ["content", "style"])
def test_cli_rejects_output_collision_before_running_engine(
    input_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    content, style = _write_inputs(tmp_path)
    output = content if input_name == "content" else style

    def unexpected_transfer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("engine should not run for a colliding output")

    monkeypatch.setattr(cli, "run_style_transfer", unexpected_transfer)

    exit_code = cli.main([str(content), str(style), "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "output path must differ" in captured.err
    assert "Traceback" not in captured.err
    assert output.is_file()


def test_cli_detects_equivalent_relative_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content, style = _write_inputs(tmp_path)
    equivalent_output = content.parent / "." / content.name
    monkeypatch.setattr(cli, "run_style_transfer", _mock_transfer)

    exit_code = cli.main([str(content), str(style), "--output", str(equivalent_output)])

    assert exit_code == 2
    assert "output path must differ" in capsys.readouterr().err


def test_cli_detects_output_hard_link_to_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content, style = _write_inputs(tmp_path)
    output = tmp_path / "content-hard-link.png"
    output.hardlink_to(content)
    monkeypatch.setattr(cli, "run_style_transfer", _mock_transfer)

    exit_code = cli.main([str(content), str(style), "--output", str(output)])

    assert exit_code == 2
    assert "output path must differ" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--learning-rate", "nan"),
        ("--content-weight", "inf"),
        ("--style-weight", "Infinity"),
        ("--tv-weight", "NaN"),
    ],
)
def test_cli_rejects_nonfinite_floats(
    option: str,
    value: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    content, style = _write_inputs(tmp_path)

    with pytest.raises(SystemExit) as error:
        cli.main([str(content), str(style), option, value])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "must be finite" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--size", "31", "must be at least 32"),
        ("--steps", "0", "must be at least 1"),
        ("--learning-rate", "0", "must be positive"),
        ("--content-weight", "-1", "cannot be negative"),
        ("--style-weight", "-1", "cannot be negative"),
        ("--tv-weight", "-1", "cannot be negative"),
    ],
)
def test_cli_rejects_out_of_domain_numbers(
    option: str,
    value: str,
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    content, style = _write_inputs(tmp_path)

    with pytest.raises(SystemExit) as error:
        cli.main([str(content), str(style), option, value])

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_reports_invalid_image_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content = tmp_path / "content.png"
    content.write_bytes(b"not an image")
    style = tmp_path / "style.png"
    Image.new("RGB", (32, 32)).save(style)

    def unexpected_transfer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("engine should not run for an invalid image")

    monkeypatch.setattr(cli, "run_style_transfer", unexpected_transfer)

    exit_code = cli.main([str(content), str(style), "--size", "32", "--steps", "1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid image" in captured.err
    assert "Traceback" not in captured.err


def test_cli_reports_missing_image_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.png"
    style = tmp_path / "style.png"
    Image.new("RGB", (32, 32)).save(style)
    monkeypatch.setattr(cli, "run_style_transfer", _mock_transfer)

    exit_code = cli.main([str(missing), str(style)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "image not found" in captured.err
    assert "Traceback" not in captured.err


def test_cli_reports_runtime_error_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content, style = _write_inputs(tmp_path)

    def unavailable_device(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("CUDA was requested but is not available")

    monkeypatch.setattr(cli, "run_style_transfer", unavailable_device)

    exit_code = cli.main(
        [str(content), str(style), "--size", "32", "--steps", "1", "--device", "cuda"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "CUDA was requested but is not available" in captured.err
    assert "Traceback" not in captured.err


def test_cli_preserve_color_uses_validated_content_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content, style = _write_inputs(tmp_path)
    output = tmp_path / "preserved.png"
    monkeypatch.setattr(cli, "run_style_transfer", _mock_transfer)
    original_loader = cli.load_pil_image
    loaded_paths: list[Path] = []

    def recording_loader(path: str | Path) -> Image.Image:
        loaded_paths.append(Path(path))
        return original_loader(path)

    monkeypatch.setattr(cli, "load_pil_image", recording_loader)

    exit_code = cli.main(
        [
            str(content),
            str(style),
            "--output",
            str(output),
            "--size",
            "32",
            "--steps",
            "1",
            "--preserve-color",
        ]
    )

    assert exit_code == 0
    assert loaded_paths == [content]
    assert output.is_file()

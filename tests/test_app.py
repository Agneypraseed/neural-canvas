from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
import torch
from PIL import Image

import app
from neural_style_transfer.engine import LossSnapshot


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    content = tmp_path / "content.png"
    style = tmp_path / "style.png"
    Image.new("RGB", (160, 96), (80, 120, 160)).save(content)
    Image.new("RGB", (96, 160), (160, 120, 80)).save(style)
    return content, style


def _snapshot(step: int) -> LossSnapshot:
    return LossSnapshot(
        step=step,
        total=1.25,
        content=0.25,
        style=0.00001,
        total_variation=0.02,
    )


@pytest.mark.parametrize(
    ("value", "name", "minimum", "maximum", "message"),
    [
        (24, "steps", 25, 100, "between 25 and 100"),
        (101, "steps", 25, 100, "between 25 and 100"),
        ("nan", "steps", 25, 100, "finite integer"),
        (25.5, "steps", 25, 100, "finite integer"),
        (True, "steps", 25, 100, "must be an integer"),
    ],
)
def test_bounded_integer_rejects_untrusted_values(
    value: object,
    name: str,
    minimum: int,
    maximum: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        app._bounded_integer(value, name=name, minimum=minimum, maximum=maximum)


@pytest.mark.parametrize("value", ["nan", "inf", 3.9, 6.1, True])
def test_bounded_float_rejects_untrusted_values(value: object) -> None:
    with pytest.raises(ValueError):
        app._bounded_float(value, name="style", minimum=4.0, maximum=6.0)


def test_public_demo_closes_api_bypass_and_bounds_resources() -> None:
    config = app.demo.get_config_file()
    render_dependency = config["dependencies"][-2]
    render_function = app.demo.fns[render_dependency["id"]]

    assert app.demo.api_open is False
    assert app.demo._queue.max_size == app.PUBLIC_QUEUE_SIZE
    assert app.demo._queue.default_concurrency_limit == 1
    assert app.demo.delete_cache == (app.CACHE_CLEANUP_SECONDS, app.CACHE_CLEANUP_SECONDS)
    assert app.LAUNCH_KWARGS["max_file_size"] == app.PUBLIC_MAX_FILE_SIZE
    assert app.LAUNCH_KWARGS["enable_monitoring"] is False
    assert render_dependency["api_visibility"] == "private"
    assert render_function.concurrency_limit == 1
    assert render_function.concurrency_id == "style-transfer"


def test_local_feature_extractor_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    created = object()
    calls: list[str] = []

    def fake_new(device_name: str) -> object:
        calls.append(device_name)
        return created

    monkeypatch.setattr(app, "_SPACES_RUNTIME", False)
    monkeypatch.setattr(app, "_new_feature_extractor", fake_new)
    app._local_feature_extractor.cache_clear()

    assert app._feature_extractor("cpu") is created
    assert app._feature_extractor("cpu") is created
    assert calls == ["cpu"]
    app._local_feature_extractor.cache_clear()


def test_hosted_feature_extractor_uses_startup_model(monkeypatch: pytest.MonkeyPatch) -> None:
    hosted = object()
    monkeypatch.setattr(app, "_SPACES_RUNTIME", True)
    monkeypatch.setattr(app, "_HOSTED_FEATURE_EXTRACTOR", hosted)
    monkeypatch.setattr(app, "_HOSTED_FEATURE_ERROR", None)

    assert app._feature_extractor("auto") is hosted


def test_local_import_uses_automatic_device_selection() -> None:
    assert app._SPACES_RUNTIME is False
    assert app.HOSTED_DEVICE == "auto"


def test_hosted_feature_extractor_reports_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("checkpoint download failed")
    monkeypatch.setattr(app, "_SPACES_RUNTIME", True)
    monkeypatch.setattr(app, "_HOSTED_FEATURE_EXTRACTOR", None)
    monkeypatch.setattr(app, "_HOSTED_FEATURE_ERROR", failure)

    with pytest.raises(RuntimeError, match="initialization did not complete") as error:
        app._feature_extractor("auto")

    assert error.value.__cause__ is failure


def test_stylize_runs_offline_with_validated_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content, style = _write_inputs(tmp_path)
    callbacks: list[int] = []

    monkeypatch.setattr(app, "_feature_extractor", lambda _device: object())

    def fake_transfer(
        content_tensor: torch.Tensor,
        _style_tensor: torch.Tensor,
        config: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        callback = kwargs["callback"]
        snapshot = _snapshot(config.steps)  # type: ignore[attr-defined]
        callback(snapshot, content_tensor)  # type: ignore[operator]
        callbacks.append(snapshot.step)
        return SimpleNamespace(
            image=content_tensor,
            history=(snapshot,),
            device="cpu",
        )

    monkeypatch.setattr(app, "run_style_transfer", fake_transfer)
    progress_updates: list[object] = []

    def progress(value: object, **_kwargs: object) -> None:
        progress_updates.append(value)

    output, summary = app.stylize(
        content,
        style,
        app.PUBLIC_DEFAULT_STEPS,
        app.PUBLIC_DEFAULT_IMAGE_SIZE,
        app.PUBLIC_DEFAULT_STYLE_EXPONENT,
        False,
        progress=progress,  # type: ignore[arg-type]
    )

    assert output.size == (128, 77)
    assert callbacks == [app.PUBLIC_DEFAULT_STEPS]
    assert progress_updates[-1] == (app.PUBLIC_DEFAULT_STEPS, app.PUBLIC_DEFAULT_STEPS)
    assert "128&times;77px" in summary
    assert "CPU" in summary


def test_stylize_rejects_server_side_limit_before_loading_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content, style = _write_inputs(tmp_path)

    def unexpected_loader(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("images must not load before parameter validation")

    monkeypatch.setattr(app, "load_pil_image", unexpected_loader)

    with pytest.raises(gr.Error, match="optimization steps must be between"):
        app.stylize(
            content,
            style,
            app.PUBLIC_MAX_STEPS + 1,
            app.PUBLIC_DEFAULT_IMAGE_SIZE,
            app.PUBLIC_DEFAULT_STYLE_EXPONENT,
            False,
            progress=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        )


def test_stylize_translates_weight_loading_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content, style = _write_inputs(tmp_path)

    def unavailable_weights(_device: str) -> None:
        raise OSError("network unavailable")

    monkeypatch.setattr(app, "_feature_extractor", unavailable_weights)

    with pytest.raises(gr.Error, match="VGG-19 weights could not be loaded"):
        app.stylize(
            content,
            style,
            app.PUBLIC_DEFAULT_STEPS,
            app.PUBLIC_DEFAULT_IMAGE_SIZE,
            app.PUBLIC_DEFAULT_STYLE_EXPONENT,
            False,
            progress=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        )

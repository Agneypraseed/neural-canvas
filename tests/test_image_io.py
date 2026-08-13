from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps, PngImagePlugin

from neural_style_transfer.image_io import (
    DEFAULT_MAX_SOURCE_PIXELS,
    MIN_IMAGE_SHORT_EDGE,
    ImageValidationError,
    load_image,
    load_pil_image,
    pil_to_tensor,
    prepare_image,
    preserve_content_colors,
    resize_long_edge,
    tensor_to_pil,
    validate_image_dimensions,
)


def _save_image(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    Image.new("RGB", size, (80, 120, 160)).save(path)


def test_resize_long_edge_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (400, 200), "white")
    resized = resize_long_edge(image, 100)
    assert resized.size == (100, 50)


def test_pil_tensor_round_trip() -> None:
    array = np.zeros((6, 8, 3), dtype=np.uint8)
    array[:, :, 0] = 128
    image = Image.fromarray(array)

    tensor = pil_to_tensor(image)
    restored = tensor_to_pil(tensor)

    assert tensor.shape == (1, 3, 6, 8)
    assert tensor.dtype == torch.float32
    assert np.max(np.abs(np.asarray(restored).astype(int) - array.astype(int))) <= 1


def test_tensor_to_pil_rejects_batches() -> None:
    with pytest.raises(ValueError):
        tensor_to_pil(torch.zeros(2, 3, 8, 8))


def test_prepare_image_accepts_formatless_trusted_image() -> None:
    prepared = prepare_image(Image.new("RGBA", (20, 16), (1, 2, 3, 255)))

    assert prepared.mode == "RGB"
    assert prepared.size == (20, 16)
    assert prepared.format is None


def test_load_pil_image_accepts_supported_format_despite_extension(tmp_path: Path) -> None:
    image_path = tmp_path / "actually-a-png.dat"
    Image.new("RGB", (32, 24), "navy").save(image_path, format="PNG")

    loaded = load_pil_image(image_path)

    assert loaded.mode == "RGB"
    assert loaded.size == (32, 24)
    assert loaded.getpixel((0, 0))[2] > 100


def test_load_pil_image_rejects_unsupported_format(tmp_path: Path) -> None:
    image_path = tmp_path / "single-frame.gif"
    Image.new("RGB", (32, 24), "white").save(image_path)

    with pytest.raises(ImageValidationError, match="unsupported image format 'GIF'"):
        load_pil_image(image_path)


def test_load_pil_image_rejects_animated_images(tmp_path: Path) -> None:
    image_path = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "blue")]
    frames[0].save(image_path, save_all=True, append_images=frames[1:], duration=20)

    with pytest.raises(ImageValidationError, match="animated images are not supported"):
        load_pil_image(image_path)


def test_load_pil_image_rejects_short_edge_below_minimum(tmp_path: Path) -> None:
    image_path = tmp_path / "too-thin.png"
    _save_image(image_path, (40, MIN_IMAGE_SHORT_EDGE - 1))

    with pytest.raises(ImageValidationError, match="shortest edge"):
        load_pil_image(image_path)


def test_load_pil_image_rejects_excessive_pixels_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "large-header.png"
    _save_image(image_path, (100, 100))

    def fail_if_loaded(self: PngImagePlugin.PngImageFile, *args: object, **kwargs: object) -> None:
        raise AssertionError("pixel data should not be loaded")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", fail_if_loaded)

    with pytest.raises(ImageValidationError, match="the limit is 9,999"):
        load_pil_image(image_path, max_source_pixels=9_999)


def test_load_pil_image_rejects_malformed_data(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.png"
    malformed_path = tmp_path / "truncated.png"
    _save_image(valid_path)
    data = valid_path.read_bytes()
    malformed_path.write_bytes(data[: len(data) // 2])

    with pytest.raises(ImageValidationError, match="could not read image|invalid image"):
        load_pil_image(malformed_path)


def test_load_pil_image_reports_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="image not found"):
        load_pil_image(missing)


def test_load_image_applies_exif_orientation_before_conversion(tmp_path: Path) -> None:
    image_path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (20, 30), "purple")
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)

    loaded = load_pil_image(image_path)
    tensor = load_image(image_path, max_size=100)

    assert loaded.size == (30, 20)
    assert tensor.shape == (1, 3, 20, 30)


def test_load_image_checks_minimum_again_after_resizing(tmp_path: Path) -> None:
    image_path = tmp_path / "extreme-aspect-ratio.png"
    _save_image(image_path, (2_000, MIN_IMAGE_SHORT_EDGE))

    with pytest.raises(ImageValidationError, match="shortest edge"):
        load_image(image_path, max_size=32)


def test_validate_image_dimensions_allows_configurable_or_disabled_limit() -> None:
    image = Image.new("RGB", (20, 20))

    with pytest.raises(ImageValidationError, match="400 pixels"):
        validate_image_dimensions(image, max_source_pixels=399)

    validate_image_dimensions(image, max_source_pixels=None)
    assert DEFAULT_MAX_SOURCE_PIXELS == 20_000_000


def test_dimension_validation_rejects_invalid_limits() -> None:
    image = Image.new("RGB", (20, 20))

    with pytest.raises(ValueError, match="max_source_pixels"):
        validate_image_dimensions(image, max_source_pixels=0)
    with pytest.raises(ValueError, match="min_short_edge"):
        validate_image_dimensions(image, min_short_edge=0)


def test_color_preservation_keeps_content_chroma() -> None:
    generated = Image.new("RGB", (8, 8), (180, 180, 180))
    content = Image.new("RGB", (8, 8), (200, 40, 30))
    result = preserve_content_colors(generated, content)

    _, result_cb, result_cr = result.convert("YCbCr").split()
    _, content_cb, content_cr = content.convert("YCbCr").split()
    _, generated_cb, generated_cr = generated.convert("YCbCr").split()

    # RGB gamut clipping can prevent an exact round trip for saturated colors,
    # but both result channels should move toward the content chroma.
    assert abs(np.asarray(result_cb).mean() - np.asarray(content_cb).mean()) < abs(
        np.asarray(generated_cb).mean() - np.asarray(content_cb).mean()
    )
    assert abs(np.asarray(result_cr).mean() - np.asarray(content_cr).mean()) < abs(
        np.asarray(generated_cr).mean() - np.asarray(content_cr).mean()
    )


def test_color_preservation_honors_content_exif_orientation() -> None:
    content = Image.new("RGB", (20, 40), (180, 80, 80))
    content.paste((80, 120, 160), (0, 20, 20, 40))
    content.getexif()[274] = 6
    oriented_content = ImageOps.exif_transpose(content).convert("RGB")
    generated = Image.new("RGB", oriented_content.size, (128, 128, 128))

    result = preserve_content_colors(generated, content)

    _, result_cb, result_cr = result.convert("YCbCr").split()
    _, expected_cb, expected_cr = oriented_content.convert("YCbCr").split()
    cb_difference = np.abs(np.asarray(result_cb, dtype=int) - np.asarray(expected_cb, dtype=int))
    cr_difference = np.abs(np.asarray(result_cr, dtype=int) - np.asarray(expected_cr, dtype=int))
    assert np.mean(cb_difference) < 2
    assert np.mean(cr_difference) < 2

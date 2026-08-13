"""Image loading, validation, conversion, saving, and color helpers."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import Tensor

DEFAULT_MAX_SOURCE_PIXELS = 20_000_000
MIN_IMAGE_SHORT_EDGE = 16
SUPPORTED_IMAGE_FORMATS = frozenset({"BMP", "JPEG", "PNG", "WEBP"})


class ImageValidationError(ValueError):
    """Raised when an image is unsafe or cannot be decoded reliably."""


def validate_image_dimensions(
    image: Image.Image,
    *,
    max_source_pixels: int | None = DEFAULT_MAX_SOURCE_PIXELS,
    min_short_edge: int = MIN_IMAGE_SHORT_EDGE,
) -> None:
    """Validate image dimensions without forcing its pixel data to decode.

    Passing ``None`` for ``max_source_pixels`` disables the pixel-count limit for
    trusted local callers. Public file loading uses the conservative default.
    """

    if max_source_pixels is not None and max_source_pixels < 1:
        raise ValueError("max_source_pixels must be positive or None")
    if min_short_edge < 1:
        raise ValueError("min_short_edge must be positive")

    width, height = image.size
    if width < 1 or height < 1:
        raise ImageValidationError("image dimensions must be positive")
    if min(width, height) < min_short_edge:
        raise ImageValidationError(
            f"image must be at least {min_short_edge} pixels along its shortest edge "
            f"(received {width}x{height})"
        )

    pixels = width * height
    if max_source_pixels is not None and pixels > max_source_pixels:
        raise ImageValidationError(
            f"image has {pixels:,} pixels; the limit is {max_source_pixels:,}"
        )


def _validate_source_image(
    image: Image.Image,
    *,
    max_source_pixels: int | None,
    min_short_edge: int,
    require_known_format: bool,
) -> None:
    """Validate source metadata before any full pixel decode."""

    try:
        frame_count = int(getattr(image, "n_frames", 1))
        is_animated = bool(getattr(image, "is_animated", False))
    except (EOFError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(f"could not inspect image frames: {exc}") from exc

    if is_animated or frame_count != 1:
        raise ImageValidationError("animated images are not supported")

    source_format = image.format.upper() if image.format else None
    if source_format is None and require_known_format:
        raise ImageValidationError("image format could not be identified")
    if source_format is not None and source_format not in SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ImageValidationError(
            f"unsupported image format {source_format!r}; supported formats: {supported}"
        )

    validate_image_dimensions(
        image,
        max_source_pixels=max_source_pixels,
        min_short_edge=min_short_edge,
    )


def _oriented_rgb(image: Image.Image) -> Image.Image:
    """Return a detached RGB image with EXIF orientation applied first."""

    try:
        oriented = ImageOps.exif_transpose(image)
        oriented.load()
        rgb = oriented.convert("RGB")
        rgb.load()
    except (EOFError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(f"could not decode image pixels: {exc}") from exc
    return rgb.copy()


def prepare_image(
    image: Image.Image,
    *,
    max_source_pixels: int | None = DEFAULT_MAX_SOURCE_PIXELS,
    min_short_edge: int = MIN_IMAGE_SHORT_EDGE,
) -> Image.Image:
    """Validate, orient, fully decode, and convert a PIL image to RGB.

    Images opened by Pillow must use one of :data:`SUPPORTED_IMAGE_FORMATS`.
    A format-less image created in memory is accepted for trusted local workflows.
    """

    _validate_source_image(
        image,
        max_source_pixels=max_source_pixels,
        min_short_edge=min_short_edge,
        require_known_format=False,
    )
    prepared = _oriented_rgb(image)
    validate_image_dimensions(
        prepared,
        max_source_pixels=max_source_pixels,
        min_short_edge=min_short_edge,
    )
    return prepared


def load_pil_image(
    path: str | Path,
    *,
    max_source_pixels: int | None = DEFAULT_MAX_SOURCE_PIXELS,
    min_short_edge: int = MIN_IMAGE_SHORT_EDGE,
) -> Image.Image:
    """Safely load a supported, single-frame image from disk as detached RGB."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    try:
        # Pillow decoding is lazy. Validate cheap header metadata and integrity on
        # one handle, then reopen for the actual decode because verify() consumes it.
        with Image.open(image_path) as candidate:
            _validate_source_image(
                candidate,
                max_source_pixels=max_source_pixels,
                min_short_edge=min_short_edge,
                require_known_format=True,
            )
            candidate.verify()

        with Image.open(image_path) as candidate:
            _validate_source_image(
                candidate,
                max_source_pixels=max_source_pixels,
                min_short_edge=min_short_edge,
                require_known_format=True,
            )
            return prepare_image(
                candidate,
                max_source_pixels=max_source_pixels,
                min_short_edge=min_short_edge,
            )
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError) as exc:
        raise ImageValidationError(f"invalid image {image_path}: {exc}") from exc
    except (EOFError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(f"could not read image {image_path}: {exc}") from exc


def resize_long_edge(image: Image.Image, max_size: int) -> Image.Image:
    """Shrink an image so its longest edge is at most ``max_size``."""

    if max_size < 1:
        raise ValueError("max_size must be positive")
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_size:
        return image.copy()
    scale = max_size / long_edge
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL image into a float tensor in the [0, 1] range."""

    array = np.asarray(_oriented_rgb(image), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    """Convert a 3D tensor or single-image batch into an RGB PIL image."""

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("a batched tensor must contain exactly one image")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("tensor must have shape (3, height, width)")

    array = (
        tensor.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array)


def load_image(
    path: str | Path,
    max_size: int,
    *,
    max_source_pixels: int | None = DEFAULT_MAX_SOURCE_PIXELS,
    min_short_edge: int = MIN_IMAGE_SHORT_EDGE,
) -> Tensor:
    """Safely load, orient, resize, and convert an image from disk."""

    image = load_pil_image(
        path,
        max_source_pixels=max_source_pixels,
        min_short_edge=min_short_edge,
    )
    resized = resize_long_edge(image, max_size)
    # Extreme aspect ratios can pass source validation but collapse below the
    # minimum when their long edge is reduced.
    validate_image_dimensions(
        resized,
        max_source_pixels=max_source_pixels,
        min_short_edge=min_short_edge,
    )
    return pil_to_tensor(resized)


def save_image(tensor: Tensor, path: str | Path) -> Path:
    """Save a tensor as an image, creating its parent directory if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(output_path)
    return output_path


def preserve_content_colors(generated: Image.Image, content: Image.Image) -> Image.Image:
    """Use generated luminance while preserving oriented content-image chroma."""

    generated_rgb = _oriented_rgb(generated)
    content_rgb = ImageOps.fit(
        _oriented_rgb(content), generated_rgb.size, method=Image.Resampling.LANCZOS
    )
    generated_y, _, _ = generated_rgb.convert("YCbCr").split()
    _, content_cb, content_cr = content_rgb.convert("YCbCr").split()
    return Image.merge("YCbCr", (generated_y, content_cb, content_cr)).convert("RGB")

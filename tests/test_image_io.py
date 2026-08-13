import numpy as np
import pytest
import torch
from PIL import Image

from neural_style_transfer.image_io import (
    pil_to_tensor,
    preserve_content_colors,
    resize_long_edge,
    tensor_to_pil,
)


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

import pytest
import torch

from neural_style_transfer.losses import (
    content_loss,
    gram_matrix,
    style_loss,
    total_variation_loss,
)


def test_gram_matrix_is_symmetric_and_normalized() -> None:
    features = torch.ones(1, 3, 2, 2)
    gram = gram_matrix(features)

    assert gram.shape == (1, 3, 3)
    assert torch.allclose(gram, gram.transpose(1, 2))
    assert torch.allclose(gram, torch.full_like(gram, 1 / 3))


def test_matching_features_have_zero_content_and_style_loss() -> None:
    features = torch.randn(1, 4, 3, 3)
    generated = {"style": features}
    targets = {"style": gram_matrix(features)}

    assert content_loss(features, features).item() == pytest.approx(0.0)
    assert style_loss(generated, targets, (("style", 1.0),)).item() == pytest.approx(0.0)


def test_total_variation_is_zero_for_constant_image() -> None:
    image = torch.full((1, 3, 8, 8), 0.5)
    assert total_variation_loss(image).item() == pytest.approx(0.0)


def test_gram_matrix_rejects_non_image_features() -> None:
    with pytest.raises(ValueError):
        gram_matrix(torch.ones(3, 4))


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param((("first", -0.5), ("second", 1.5)), id="negative-member"),
        pytest.param((("first", float("nan")),), id="nan"),
        pytest.param((("first", float("inf")),), id="positive-infinity"),
        pytest.param((("first", float("-inf")),), id="negative-infinity"),
        pytest.param((("first", 0.0), ("second", 0.0)), id="all-zero"),
    ],
)
def test_style_loss_rejects_invalid_layer_weights(
    weights: tuple[tuple[str, float], ...],
) -> None:
    features = {
        "first": torch.ones(1, 1, 2, 2),
        "second": torch.ones(1, 1, 2, 2),
    }
    targets = {name: gram_matrix(value) for name, value in features.items()}

    with pytest.raises(ValueError, match="layer weights"):
        style_loss(features, targets, weights)


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param(torch.tensor([[[[0.0, 1.0, 3.0]]]]), 1.5, id="one-by-n"),
        pytest.param(torch.tensor([[[[0.0], [1.0], [3.0]]]]), 1.5, id="n-by-one"),
        pytest.param(torch.tensor([[[[2.0]]]], requires_grad=True), 0.0, id="one-by-one"),
    ],
)
def test_total_variation_handles_singleton_spatial_dimensions(
    image: torch.Tensor, expected: float
) -> None:
    loss = total_variation_loss(image)

    assert loss.item() == pytest.approx(expected)
    assert torch.isfinite(loss)
    if image.requires_grad:
        loss.backward()
        assert torch.equal(image.grad, torch.zeros_like(image))


def test_total_variation_rejects_empty_spatial_dimensions() -> None:
    with pytest.raises(ValueError, match="spatial dimensions must be positive"):
        total_variation_loss(torch.empty(1, 3, 0, 4))

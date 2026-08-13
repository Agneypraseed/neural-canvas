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

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from neural_style_transfer.config import StyleTransferConfig
from neural_style_transfer.engine import run_style_transfer
from neural_style_transfer.losses import (
    content_loss,
    gram_matrix,
    style_loss,
    total_variation_loss,
)


class TinyFeatureExtractor(nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "style": image,
            "content": F.avg_pool2d(image, kernel_size=2),
        }


def test_complete_optimization_loop_runs_offline() -> None:
    content = torch.zeros(1, 3, 8, 8)
    content[:, :, 2:6, 2:6] = 0.8
    style = torch.rand(1, 3, 6, 6)
    config = StyleTransferConfig(
        image_size=32,
        steps=3,
        learning_rate=0.05,
        content_layer="content",
        style_layers=(("style", 1.0),),
        style_weight=10,
        total_variation_weight=0.01,
        device="cpu",
        progress_interval=1,
    )

    result = run_style_transfer(content, style, config, feature_extractor=TinyFeatureExtractor())

    assert result.image.shape == content.shape
    assert result.device == "cpu"
    assert len(result.history) == 3
    assert torch.all((result.image >= 0) & (result.image <= 1))
    assert all(torch.isfinite(torch.tensor(item.total)) for item in result.history)


def test_reported_losses_describe_the_returned_post_update_image() -> None:
    content = torch.zeros(1, 3, 4, 4)
    content[:, :, 1:3, 1:3] = 0.8
    style = torch.linspace(0, 1, 48).reshape(1, 3, 4, 4)
    extractor = TinyFeatureExtractor()
    config = StyleTransferConfig(
        image_size=32,
        steps=1,
        learning_rate=0.1,
        content_layer="content",
        style_layers=(("style", 1.0),),
        content_weight=1.5,
        style_weight=10.0,
        total_variation_weight=0.2,
        device="cpu",
        progress_interval=1,
    )
    callbacks: list[tuple[object, torch.Tensor]] = []

    result = run_style_transfer(
        content,
        style,
        config,
        feature_extractor=extractor,
        callback=lambda snapshot, image: callbacks.append((snapshot, image)),
    )

    with torch.no_grad():
        content_target = extractor(content)["content"]
        style_targets = {"style": gram_matrix(extractor(style)["style"])}
        generated_features = extractor(result.image)
        expected_content = content_loss(generated_features["content"], content_target)
        expected_style = style_loss(generated_features, style_targets, config.style_layers)
        expected_tv = total_variation_loss(result.image)
        expected_total = (
            config.content_weight * expected_content
            + config.style_weight * expected_style
            + config.total_variation_weight * expected_tv
        )

    snapshot = result.history[-1]
    assert snapshot.content == pytest.approx(expected_content.item())
    assert snapshot.style == pytest.approx(expected_style.item())
    assert snapshot.total_variation == pytest.approx(expected_tv.item())
    assert snapshot.total == pytest.approx(expected_total.item())
    assert callbacks[0][0] == snapshot
    assert torch.equal(callbacks[0][1], result.image)


@pytest.mark.parametrize("invalid_input", ["content", "style"])
@pytest.mark.parametrize(
    "non_finite",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_engine_rejects_non_finite_image_tensors(invalid_input: str, non_finite: float) -> None:
    content = torch.zeros(1, 3, 4, 4)
    style = torch.ones(1, 3, 4, 4)
    (content if invalid_input == "content" else style)[0, 0, 0, 0] = non_finite
    config = StyleTransferConfig(
        image_size=32,
        steps=1,
        content_layer="content",
        style_layers=(("style", 1.0),),
        device="cpu",
    )

    with pytest.raises(ValueError, match=f"{invalid_input}_image must contain only finite"):
        run_style_transfer(
            content,
            style,
            config,
            feature_extractor=TinyFeatureExtractor(),
        )

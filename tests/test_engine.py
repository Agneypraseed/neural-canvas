import torch
from torch import nn
from torch.nn import functional as F

from neural_style_transfer.config import StyleTransferConfig
from neural_style_transfer.engine import run_style_transfer


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

    result = run_style_transfer(
        content, style, config, feature_extractor=TinyFeatureExtractor()
    )

    assert result.image.shape == content.shape
    assert result.device == "cpu"
    assert len(result.history) == 3
    assert torch.all((result.image >= 0) & (result.image <= 1))
    assert all(torch.isfinite(torch.tensor(item.total)) for item in result.history)

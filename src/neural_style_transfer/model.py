"""Feature extraction using an ImageNet-pretrained VGG-19 network."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19

VGG_LAYER_NAMES = {
    1: "relu1_1",
    3: "relu1_2",
    6: "relu2_1",
    8: "relu2_2",
    11: "relu3_1",
    13: "relu3_2",
    15: "relu3_3",
    17: "relu3_4",
    20: "relu4_1",
    22: "relu4_2",
    24: "relu4_3",
    26: "relu4_4",
    29: "relu5_1",
    31: "relu5_2",
    33: "relu5_3",
    35: "relu5_4",
}


class VGG19FeatureExtractor(nn.Module):
    """Expose named intermediate VGG-19 activations."""

    def __init__(
        self,
        requested_layers: Iterable[str],
        weights: VGG19_Weights = VGG19_Weights.DEFAULT,
    ) -> None:
        super().__init__()
        requested = frozenset(requested_layers)
        known = frozenset(VGG_LAYER_NAMES.values())
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown VGG layers: {', '.join(sorted(unknown))}")
        if not requested:
            raise ValueError("at least one VGG layer must be requested")

        pretrained = vgg19(weights=weights).features.eval()
        layers: list[nn.Module] = []
        for layer in pretrained.children():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            layers.append(layer)

        self.features = nn.Sequential(*layers)
        self.requested_layers = requested
        self.last_required_index = max(
            index for index, name in VGG_LAYER_NAMES.items() if name in requested
        )
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (batch, 3, height, width)")

        activation = (image - self.mean) / self.std
        outputs: dict[str, Tensor] = {}
        for index, layer in enumerate(self.features):
            activation = layer(activation)
            layer_name = VGG_LAYER_NAMES.get(index)
            if layer_name in self.requested_layers:
                outputs[layer_name] = activation
            if index >= self.last_required_index:
                break
        return outputs

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import neural_style_transfer.model as model_module
from neural_style_transfer.model import VGG_LAYER_NAMES, VGG19FeatureExtractor


def _lightweight_vgg19(*, weights: object) -> SimpleNamespace:
    del weights
    layers: list[nn.Module] = [nn.Identity() for _ in range(36)]
    for index in VGG_LAYER_NAMES:
        layers[index] = nn.ReLU(inplace=True)
    for index in (4, 9, 18, 27):
        layers[index] = nn.MaxPool2d(kernel_size=2, stride=2)
    return SimpleNamespace(features=nn.Sequential(*layers))


def test_vgg_short_edge_validation_tracks_requested_pooling_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_module, "vgg19", _lightweight_vgg19)
    deep_extractor = VGG19FeatureExtractor({"relu5_1"}, weights=None)
    shallow_extractor = VGG19FeatureExtractor({"relu1_1"}, weights=None)

    assert deep_extractor.minimum_input_size == 16
    with pytest.raises(ValueError, match="at least 16 pixels.*got 15x16"):
        deep_extractor(torch.rand(1, 3, 15, 16))
    assert deep_extractor(torch.rand(1, 3, 16, 16))["relu5_1"].shape[-2:] == (1, 1)

    assert shallow_extractor.minimum_input_size == 1
    assert shallow_extractor(torch.rand(1, 3, 1, 1))["relu1_1"].shape[-2:] == (1, 1)


def test_real_vgg_relu5_boundary_without_downloading_weights() -> None:
    extractor = VGG19FeatureExtractor({"relu5_1"}, weights=None)

    with pytest.raises(ValueError, match="at least 16 pixels.*got 15x16"):
        extractor(torch.rand(1, 3, 15, 16))
    assert extractor(torch.rand(1, 3, 16, 16))["relu5_1"].shape[-2:] == (1, 1)

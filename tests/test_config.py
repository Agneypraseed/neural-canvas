import pytest

from neural_style_transfer.config import StyleTransferConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_size", 16),
        ("steps", 0),
        ("learning_rate", 0),
        ("style_weight", -1),
        ("initialization", "invalid"),
        ("device", "tpu"),
    ],
)
def test_invalid_config_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        StyleTransferConfig(**{field: value})


def test_default_style_weights_sum_to_one() -> None:
    config = StyleTransferConfig()
    assert sum(weight for _, weight in config.style_layers) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field",
    ["learning_rate", "content_weight", "style_weight", "total_variation_weight"],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_config_rejects_non_finite_numeric_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        StyleTransferConfig(**{field: value})


@pytest.mark.parametrize(
    "weight",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_config_rejects_non_finite_style_layer_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        StyleTransferConfig(style_layers=(("style", weight),))

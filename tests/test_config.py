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

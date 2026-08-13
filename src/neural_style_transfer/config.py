"""Validated configuration for a neural-style-transfer run."""

from dataclasses import dataclass

DEFAULT_STYLE_LAYERS = (
    ("relu1_1", 0.20),
    ("relu2_1", 0.20),
    ("relu3_1", 0.20),
    ("relu4_1", 0.20),
    ("relu5_1", 0.20),
)


@dataclass(frozen=True, slots=True)
class StyleTransferConfig:
    """Hyperparameters and runtime settings for style transfer."""

    image_size: int = 512
    steps: int = 300
    learning_rate: float = 0.02
    content_weight: float = 1.0
    style_weight: float = 100_000.0
    total_variation_weight: float = 0.0001
    content_layer: str = "relu4_2"
    style_layers: tuple[tuple[str, float], ...] = DEFAULT_STYLE_LAYERS
    initialization: str = "content"
    device: str = "auto"
    seed: int = 42
    progress_interval: int = 25

    def __post_init__(self) -> None:
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.content_weight < 0 or self.style_weight < 0:
            raise ValueError("content_weight and style_weight cannot be negative")
        if self.total_variation_weight < 0:
            raise ValueError("total_variation_weight cannot be negative")
        if not self.content_layer:
            raise ValueError("content_layer cannot be empty")
        if not self.style_layers:
            raise ValueError("at least one style layer is required")
        if any(not name or weight < 0 for name, weight in self.style_layers):
            raise ValueError("style layers need a name and a non-negative weight")
        if sum(weight for _, weight in self.style_layers) <= 0:
            raise ValueError("style layer weights must have a positive sum")
        if self.initialization not in {"content", "noise"}:
            raise ValueError("initialization must be 'content' or 'noise'")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be at least 1")

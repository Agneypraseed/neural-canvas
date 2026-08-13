"""Command-line interface for Neural Canvas."""

import argparse
from pathlib import Path

from PIL import Image

from neural_style_transfer.config import StyleTransferConfig
from neural_style_transfer.engine import LossSnapshot, run_style_transfer
from neural_style_transfer.image_io import (
    load_image,
    preserve_content_colors,
    save_image,
    tensor_to_pil,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nst",
        description="Combine the structure of one image with the visual style of another.",
    )
    parser.add_argument("content", type=Path, help="path to the content image")
    parser.add_argument("style", type=Path, help="path to the style image")
    parser.add_argument(
        "--output", type=Path, default=Path("output/stylized.png"), help="output image path"
    )
    parser.add_argument("--size", type=int, default=512, help="maximum content-image edge")
    parser.add_argument("--steps", type=int, default=300, help="optimization steps")
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=100_000.0)
    parser.add_argument("--tv-weight", type=float, default=0.0001)
    parser.add_argument("--initialization", choices=("content", "noise"), default="content")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preserve-color",
        action="store_true",
        help="preserve the content image's color channels",
    )
    return parser


def _print_progress(snapshot: LossSnapshot, _image: object) -> None:
    print(
        f"step {snapshot.step:>4} | total {snapshot.total:.5f} | "
        f"content {snapshot.content:.5f} | style {snapshot.style:.5f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StyleTransferConfig(
        image_size=args.size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        total_variation_weight=args.tv_weight,
        initialization=args.initialization,
        device=args.device,
        seed=args.seed,
        progress_interval=max(1, args.steps // 10),
    )

    content = load_image(args.content, config.image_size)
    style = load_image(args.style, config.image_size)
    result = run_style_transfer(content, style, config, callback=_print_progress)

    if args.preserve_color:
        generated = tensor_to_pil(result.image)
        with Image.open(args.content) as original_content:
            generated = preserve_content_colors(generated, original_content)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generated.save(args.output)
    else:
        save_image(result.image, args.output)

    print(f"saved {args.output} using {result.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

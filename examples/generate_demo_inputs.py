"""Create deterministic, redistribution-safe demo inputs for Neural Canvas."""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = Path(__file__).parent
WIDTH = 384
HEIGHT = 256


def _gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()
    for y in range(HEIGHT):
        ratio = y / (HEIGHT - 1)
        color = tuple(
            round(a * (1 - ratio) + b * ratio) for a, b in zip(top, bottom, strict=True)
        )
        for x in range(WIDTH):
            pixels[x, y] = color
    return image


def create_content_image() -> Image.Image:
    """Draw a compact Berlin-inspired riverside scene."""

    image = _gradient((117, 177, 214), (239, 198, 150))
    draw = ImageDraw.Draw(image)

    draw.ellipse((286, 24, 330, 68), fill=(247, 210, 128))
    draw.polygon(((0, 132), (384, 119), (384, 182), (0, 190)), fill=(78, 85, 88))
    draw.rectangle((0, 184, 384, 256), fill=(52, 107, 129))

    buildings = [
        (0, 100, 58, 174, (154, 112, 93)),
        (50, 88, 112, 174, (197, 163, 127)),
        (105, 111, 170, 174, (130, 104, 96)),
        (286, 96, 350, 174, (190, 145, 112)),
        (342, 108, 384, 174, (130, 98, 91)),
    ]
    for left, top, right, bottom, color in buildings:
        draw.rectangle((left, top, right, bottom), fill=color)
        for window_y in range(top + 12, bottom - 8, 18):
            for window_x in range(left + 9, right - 6, 15):
                draw.rectangle(
                    (window_x, window_y, window_x + 6, window_y + 9),
                    fill=(224, 199, 145),
                )

    # Fernsehturm-inspired silhouette.
    draw.rectangle((226, 65, 231, 174), fill=(55, 61, 66))
    draw.polygon(((228, 15), (224, 70), (233, 70)), fill=(55, 61, 66))
    draw.ellipse((208, 52, 249, 93), fill=(178, 184, 183), outline=(49, 57, 63), width=3)
    draw.rectangle((215, 76, 242, 82), fill=(58, 65, 70))

    # Arched bridge and its reflection.
    draw.rectangle((0, 160, 384, 173), fill=(82, 66, 61))
    for center in (52, 151, 250, 349):
        draw.ellipse((center - 38, 145, center + 38, 205), fill=(82, 66, 61))
        draw.ellipse((center - 29, 155, center + 29, 207), fill=(52, 107, 129))
    for x in range(0, WIDTH, 20):
        draw.line((x, 219, min(WIDTH, x + 35), 219), fill=(118, 157, 165), width=2)
        draw.line((x + 8, 237, min(WIDTH, x + 28), 237), fill=(38, 84, 110), width=2)

    return image


def create_style_image(seed: int = 7) -> Image.Image:
    """Draw a textured geometric painting with an original palette."""

    random.seed(seed)
    image = Image.new("RGB", (WIDTH, HEIGHT), (231, 215, 178))
    draw = ImageDraw.Draw(image, "RGBA")
    palette = [
        (25, 61, 80, 220),
        (210, 76, 51, 210),
        (232, 166, 48, 210),
        (58, 121, 113, 210),
        (102, 73, 121, 185),
    ]

    for _ in range(34):
        x = random.randint(-40, WIDTH + 20)
        y = random.randint(-30, HEIGHT + 20)
        radius = random.randint(14, 66)
        color = random.choice(palette)
        if random.random() < 0.5:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        else:
            angle = random.random() * math.tau
            points = []
            for offset in (0, math.tau / 3, 2 * math.tau / 3):
                points.append(
                    (
                        x + math.cos(angle + offset) * radius,
                        y + math.sin(angle + offset) * radius,
                    )
                )
            draw.polygon(points, fill=color)

    for _ in range(90):
        x = random.randrange(WIDTH)
        y = random.randrange(HEIGHT)
        length = random.randint(8, 42)
        color = random.choice(palette)
        draw.line((x, y, x + length, y + random.randint(-8, 8)), fill=color, width=2)

    texture = (
        Image.effect_noise((WIDTH, HEIGHT), 28)
        .convert("L")
        .filter(ImageFilter.GaussianBlur(0.5))
    )
    texture_color = Image.new("RGB", (WIDTH, HEIGHT), (91, 70, 56))
    image = Image.composite(texture_color, image, texture.point(lambda value: value // 8))
    return image


def main() -> None:
    create_content_image().save(OUTPUT_DIR / "content.png")
    create_style_image().save(OUTPUT_DIR / "style.png")
    print(f"wrote demo inputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

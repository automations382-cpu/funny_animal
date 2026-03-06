"""
generate_watermark.py — One-time script to generate the watermark PNG.
Run once: python funny_animal/generate_watermark.py
Creates: funny_animal/assets/watermark.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path(__file__).parent / "assets" / "watermark.png"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Canvas: transparent background, sized for text
W, H = 420, 90
img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

TEXT = "@FunnyAnimals"

# Try to use a bold system font, fall back to default
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
except OSError:
    font = ImageFont.load_default()

# Shadow (dark, semi-transparent) for readability
for offset in [(3, 3), (2, 2)]:
    draw.text(offset, TEXT, font=font, fill=(0, 0, 0, 160))

# Main white text with 85% opacity
draw.text((0, 0), TEXT, font=font, fill=(255, 255, 255, 217))

img.save(OUTPUT, "PNG")
print(f"✓ Watermark saved: {OUTPUT}")

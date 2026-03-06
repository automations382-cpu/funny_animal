"""
card_generator.py — White Meme Card Generator
Generates a 1080x1920 white background image with:
  - Caption text rendered at the top using Impact (bold) font
  - Full emoji support via NotoColorEmoji font (CBDT/CBLC tables)
  - @madanimalx branding at the bottom
  - A centered "video slot" rectangle showing where the video will be placed
Saves the card as a PNG to .tmp/processed/meme_card.png
"""

import os
import re
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FONTS_DIR = BASE_DIR / "assets" / "fonts"
PROCESSED_DIR = BASE_DIR / ".tmp" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Canvas dimensions
CARD_W = 1080
CARD_H = 1920
CARD_BGCOLOR = (255, 255, 255)  # pure white

# Layout constants
PADDING = 50
TOP_TEXT_Y = 60              # y-start for caption text block
BRAND_BOTTOM_MARGIN = 60     # px from bottom for @madanimalx
BRAND_TEXT = "@madanimalx"

# Font fallback chain
IMPACT_CANDIDATES = [
    FONTS_DIR / "Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # fallback
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
EMOJI_CANDIDATES = [
    FONTS_DIR / "NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto-color-emoji/NotoColorEmoji.ttf",
]


def _load_font(candidates: list, size: int) -> ImageFont.FreeTypeFont:
    """Try each font path in order, return the first one that loads."""
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except (OSError, IOError):
            continue
    print("[card_generator] WARNING: No custom font found, using PIL default")
    return ImageFont.load_default()


def _is_emoji(char: str) -> bool:
    """Rough emoji detection by Unicode range."""
    cp = ord(char)
    return (
        0x1F300 <= cp <= 0x1FAFF   # Miscellaneous Symbols, Emoticons, etc.
        or 0x2600 <= cp <= 0x26FF   # Misc symbols
        or 0x2700 <= cp <= 0x27BF   # Dingbats
        or 0xFE00 <= cp <= 0xFE0F   # Variation selectors
    )


def _split_text_emoji(text: str) -> list[tuple[str, bool]]:
    """
    Split a string into segments of (segment_text, is_emoji).
    Groups consecutive characters of the same type together.
    Example: "LOL 😂🔥 wow" → [("LOL ", False), ("😂🔥", True), (" wow", False)]
    """
    if not text:
        return []
    segments = []
    current = text[0]
    current_is_emoji = _is_emoji(text[0])
    for char in text[1:]:
        char_is_emoji = _is_emoji(char)
        if char_is_emoji == current_is_emoji:
            current += char
        else:
            segments.append((current, current_is_emoji))
            current = char
            current_is_emoji = char_is_emoji
    segments.append((current, current_is_emoji))
    return segments


def draw_mixed_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    text_font: ImageFont.FreeTypeFont,
    emoji_font: ImageFont.FreeTypeFont,
    fill: tuple = (20, 20, 20),
    max_width: int = 980,
) -> int:
    """
    Render a string that may contain both regular text and emojis.
    Falls back to text-only rendering if emoji font isn't available.
    Returns the y-coordinate after the last rendered line.

    Strategy:
      - Word-wrap the text to fit max_width
      - For each line, split into text/emoji segments
      - Render segments side-by-side using appropriate font
    """
    x_start, y = xy
    line_height = text_font.size + 14  # extra leading

    # Word-wrap
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test = (current_line + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=text_font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines:
        x = x_start
        segments = _split_text_emoji(line)
        for seg_text, is_emoji_seg in segments:
            font = emoji_font if is_emoji_seg else text_font
            if font is None:
                font = text_font
            # Draw shadow for readability
            draw.text((x + 2, y + 2), seg_text, font=font, fill=(0, 0, 0, 80))
            draw.text((x, y), seg_text, font=font, fill=fill)
            bbox = draw.textbbox((x, y), seg_text, font=font)
            x = bbox[2] + 4  # advance x by the width of this segment
        y += line_height

    return y  # returns the y position after all lines


def generate_card(
    caption: str,
    output_path: Path | None = None,
    video_slot_height: int = 800,
) -> tuple[Path, tuple[int, int, int, int]]:
    """
    Generate a 1080x1920 White Meme Card PNG.

    Args:
        caption: The full caption text (may include emojis).
        output_path: Where to save the PNG. Defaults to .tmp/processed/meme_card.png
        video_slot_height: Height in pixels reserved for the video overlay.

    Returns:
        (path_to_png, video_slot_rect) where video_slot_rect = (x, y, w, h)
        describing where the video should be overlaid by FFmpeg/MoviePy.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "meme_card.png"

    # ── Create canvas ────────────────────────────────────────────
    img = Image.new("RGBA", (CARD_W, CARD_H), CARD_BGCOLOR + (255,))
    draw = ImageDraw.Draw(img)

    # ── Fonts ────────────────────────────────────────────────────
    caption_font = _load_font(IMPACT_CANDIDATES, 68)
    emoji_font = _load_font(EMOJI_CANDIDATES, 60)
    brand_font = _load_font(IMPACT_CANDIDATES, 52)

    # ── Parse caption into 4 lines ───────────────────────────────
    # Expected format:
    #   Line 0: Headline (ALL CAPS)
    #   Line 1: Meme joke
    #   Line 2: Emojis
    #   Line 3: Hashtags
    lines = caption.strip().split("\n")
    headline = lines[0] if len(lines) > 0 else "FUNNY ANIMAL ALERT"
    joke = lines[1] if len(lines) > 1 else ""
    emojis = lines[2] if len(lines) > 2 else ""
    hashtags = lines[3] if len(lines) > 3 else ""

    # ── Draw headline (large, black) ─────────────────────────────
    y = TOP_TEXT_Y
    y = draw_mixed_text(
        draw, headline, (PADDING, y),
        text_font=caption_font, emoji_font=emoji_font,
        fill=(10, 10, 10), max_width=CARD_W - PADDING * 2
    )
    y += 10  # small gap

    # ── Draw joke line (medium, dark gray) ───────────────────────
    joke_font = _load_font(IMPACT_CANDIDATES, 52)
    y = draw_mixed_text(
        draw, joke, (PADDING, y),
        text_font=joke_font, emoji_font=emoji_font,
        fill=(40, 40, 40), max_width=CARD_W - PADDING * 2
    )
    y += 10

    # ── Draw emoji line ──────────────────────────────────────────
    y = draw_mixed_text(
        draw, emojis, (PADDING, y),
        text_font=joke_font, emoji_font=emoji_font,
        fill=(30, 30, 30), max_width=CARD_W - PADDING * 2
    )
    y += 20  # gap before video slot

    # ── Video slot (centered rectangle, light gray placeholder) ──
    slot_x = 0
    slot_y = y
    slot_w = CARD_W
    slot_h = video_slot_height

    # Draw a subtle placeholder rectangle (will be overwritten by actual video)
    draw.rectangle(
        [slot_x, slot_y, slot_x + slot_w, slot_y + slot_h],
        fill=(230, 230, 230), outline=(200, 200, 200), width=2
    )
    video_slot = (slot_x, slot_y, slot_w, slot_h)

    # ── Hashtags below video slot ─────────────────────────────────
    ht_y = slot_y + slot_h + 20
    ht_font = _load_font(IMPACT_CANDIDATES, 34)
    if hashtags:
        draw_mixed_text(
            draw, hashtags, (PADDING, ht_y),
            text_font=ht_font, emoji_font=emoji_font,
            fill=(80, 80, 200), max_width=CARD_W - PADDING * 2
        )

    # ── Brand watermark at very bottom ───────────────────────────
    brand_bbox = draw.textbbox((0, 0), BRAND_TEXT, font=brand_font)
    brand_w = brand_bbox[2] - brand_bbox[0]
    brand_x = (CARD_W - brand_w) // 2
    brand_y = CARD_H - BRAND_BOTTOM_MARGIN - (brand_bbox[3] - brand_bbox[1])

    # Shadow
    draw.text((brand_x + 2, brand_y + 2), BRAND_TEXT, font=brand_font, fill=(0, 0, 0, 120))
    draw.text((brand_x, brand_y), BRAND_TEXT, font=brand_font, fill=(30, 30, 30))

    # ── Save ──────────────────────────────────────────────────────
    img = img.convert("RGB")
    img.save(str(output_path), "PNG", optimize=False)
    print(f"[card_generator] ✓ Card saved → {output_path} | video_slot={video_slot}")
    return output_path, video_slot


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    sample_caption = args.caption or (
        "WHEN YOUR CAT JUDGES YOU 24/7\n"
        "Me trying to be productive vs my cat:\n"
        "😂🐱🔥💀👀\n"
        "#funnycat #catmemes #animalmemes #reels #funnypets #viral #trending"
    )
    out = Path(args.out) if args.out else None
    path, slot = generate_card(sample_caption, output_path=out)
    print(f"Card: {path}  |  Video slot: {slot}")

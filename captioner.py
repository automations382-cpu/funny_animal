"""
captioner.py — Caption Generation Module
Uses Google Gemini Flash (free tier) to generate short meme captions.
Saves caption as {video_filename}.txt
"""

import os
import sys
import argparse
from pathlib import Path

import google.genai as genai
from google.genai import types as genai_types


CAPTION_PROMPT = """You are a viral Instagram Reels caption writer for a funny animal memes channel.

Write a caption for a funny animal video. It must follow this EXACT format:

Line 1: A short punchy ALL-CAPS heading (max 6 words, no hashtags)
Line 2: A funny one-liner meme caption about animals (max 12 words)
Line 3: A string of 5-7 relevant emojis only
Line 4: 15 relevant Instagram hashtags separated by spaces

Rules:
- Keep it brief and punchy — don't over-generate
- The joke should feel relatable and shareable
- Mix animal + meme + viral hashtags
- No emojis on other lines except Line 3
- Output ONLY the 4 lines, no explanations

Example output:
WHEN YOUR CAT JUDGES YOU 24/7
Me: "it's fine" My cat at 3am:
🐱😂🔥👀💀
#funnycats #catmemes #animalmemes #catsofinstagram #funnyanimals #catvideos #reels #viralreels #catlovers #memes #animalsoftiktok #funnypets #funnymemes #instagramreels #catlife"""


def generate_caption(gemini_key: str) -> str:
    """
    Call Gemini Flash API and return the caption string.
    Falls back to a static caption if API call fails.
    """
    try:
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=CAPTION_PROMPT,
            config=genai_types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=200,
            ),
        )
        caption = response.text.strip()
        print("[captioner] ✓ Caption generated via Gemini")
        return caption
    except Exception as e:
        print(f"[captioner] Gemini error: {e} — using fallback caption")
        return fallback_caption()


def fallback_caption() -> str:
    """Static fallback caption if Gemini is unavailable."""
    import random
    captions = [
        "ANIMALS BEING CHAOTIC LEGENDS\nMe trying to be productive vs my pet:\n😂🐾🔥💀🐶\n#funnyanimals #animalmemes #funnypets #viralpets #reels #funnymemes #animalsoftiktok #petlovers #dogmemes #catmemes #laughoutloud #moodbooster #instagramreels #funnyvideos #petlife",
        "WHEN ANIMALS HAVE ZERO CHILL\nNobody:\nMy dog at 3am:\n🐕💀😂🔥⚡\n#dogmemes #funnydog #animalsofinstagram #dogvideos #reels #viralmemes #funnypets #dogsofinstagram #petmemes #funnyreels #catvideos #animalmemes #laughing #trending #funnyanimal",
        "POV: YOU HAVE A FUNNY PET\nExpectation vs reality of owning a pet:\n🐱😭😂🐾🎭\n#funnycats #catvideos #petlife #funnyanimals #catmemes #reels #animalmemes #viralreels #catsoftiktok #funnypets #catlovers #memes #instagramreels #humor #trending",
    ]
    return random.choice(captions)


def save_caption(caption: str, video_path: Path) -> Path:
    """Save caption as a .txt file with the same stem as the video."""
    txt_path = video_path.with_suffix(".txt")
    txt_path.write_text(caption, encoding="utf-8")
    print(f"[captioner] Caption saved → {txt_path.name}")
    return txt_path


# ── CLI test mode ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("[captioner] No GEMINI_API_KEY — showing fallback")
        caption = fallback_caption()
    else:
        caption = generate_caption(gemini_key)

    print("\n--- CAPTION ---")
    print(caption)

    if args.video:
        save_caption(caption, Path(args.video))

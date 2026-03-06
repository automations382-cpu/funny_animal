"""
ai_director.py — AI-Driven Video Editing Engine (v2)

Uses Gemini 1.5 Flash to analyze each raw animal clip and generate a precise
editing plan (punch text, timestamps, meme insert point). Then uses MoviePy
to assemble the final reel on a White Meme Card template.

Architecture:
  1. Upload raw video to Gemini Files API
  2. Prompt Gemini to return a strict JSON editing plan
  3. Use MoviePy to:
     a. Create 1080x1920 white canvas background
     b. Place the video centered in the canvas
     c. Render punchline text (Impact font) at Gemini-specified timestamps
     d. Split video at meme_insert_timestamp, insert meme clip, resume
     e. Burn @madanimalx branding at the bottom
  4. Concatenate all assembled clips into the final reel

Graceful fallback: if Gemini returns invalid JSON or the API call fails,
the clip is assembled without mid-clip meme splits (standard path).
"""

import os
import sys
import json
import time
import random
import shutil
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import google.genai as genai
from google.genai import types as genai_types
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    TextClip,
    ColorClip,
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MEME_DIR = BASE_DIR / "meme_inserts"
FONTS_DIR = BASE_DIR / "assets" / "fonts"
PROCESSED_DIR = BASE_DIR / ".tmp" / "processed"
OUTPUT_DIR = BASE_DIR / ".tmp" / "output"

for d in [PROCESSED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Canvas
CARD_W, CARD_H = 1080, 1920
BRAND_TEXT = "@madanimalx"
MAX_CLIP_DURATION = 10.0     # hard cap per source clip (seconds)
MEME_DURATION = 3.0          # max seconds to take from each meme insert

# Font paths (fallback chain)
IMPACT_CANDIDATES = [
    FONTS_DIR / "Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# ── Gemini prompt ──────────────────────────────────────────────────────────────
DIRECTOR_PROMPT = """You are a viral Instagram Reels editor specializing in funny animal content.

Analyze the uploaded animal video carefully. You must return ONLY a single valid JSON object — no markdown fences, no extra text.

Fields required:
{
  "punchline_text": "Short funny caption with 1-2 emojis (max 8 words)",
  "text_start_time": <float, seconds when text should appear>,
  "text_end_time": <float, seconds when text should disappear>,
  "meme_insert_timestamp": <float, exact second to pause video and insert meme>,
  "meme_filename": "One of: doge_dance.mp4 | cat_laugh.mp4 | dog_surprised.mp4 | cat_piano.mp4 | dog_fail.mp4"
}

Rules:
- text timings must be within video duration
- meme_insert_timestamp must be > 1.0 and < (video_duration - 1.0)
- text_end_time must be > text_start_time
- punchline_text must be funny and match the action in the video
- choose the meme_filename that best fits the moment

Return ONLY the JSON object."""


# ── Helper: find best available font ─────────────────────────────────────────

def _find_font(size: int = 60) -> str:
    """Return path string to the best available Impact-style font."""
    for p in IMPACT_CANDIDATES:
        if Path(p).exists():
            return str(p)
    return ""   # MoviePy will use default if empty


def _make_white_card() -> np.ndarray:
    """
    Return a numpy array (H×W×3) of a pure white 1080x1920 frame.
    MoviePy ImageClip accepts numpy arrays directly.
    """
    return np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)


# ── Step 1: Gemini video analysis ─────────────────────────────────────────────

def analyse_with_gemini(
    video_path: Path,
    gemini_key: str,
    retries: int = 2,
) -> Optional[dict]:
    """
    Upload video_path to Gemini Files API, send the director prompt,
    and return the parsed JSON editing plan.

    Wrapper around the new google-genai SDK (google.genai.Client).
    Uploads video to Gemini Files API, sends the director prompt,
    and returns the parsed JSON editing plan.
    Returns None on failure (triggers fallback path).
    """
    client = genai.Client(api_key=gemini_key)

    for attempt in range(retries + 1):
        try:
            print(f"[ai_director] Uploading {video_path.name} to Gemini Files API ...")
            # Upload file using Files API
            uploaded = client.files.upload(
                path=str(video_path),
                config={"mime_type": "video/mp4", "display_name": video_path.stem},
            )

            # Wait for file to become ACTIVE
            for _ in range(20):
                file_info = client.files.get(name=uploaded.name)
                if file_info.state.name == "ACTIVE":
                    break
                print("[ai_director] Waiting for Gemini file processing ...")
                time.sleep(3)
            else:
                print("[ai_director] File never became ACTIVE — skipping")
                return None

            print("[ai_director] Sending analysis prompt ...")
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[uploaded, DIRECTOR_PROMPT],
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                ),
            )

            # Clean up uploaded file
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

            raw_text = response.text.strip()
            # Strip any accidental markdown fences
            raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            plan = json.loads(raw_text)
            # Validate required keys
            required = {"punchline_text", "text_start_time", "text_end_time",
                       "meme_insert_timestamp", "meme_filename"}
            if not required.issubset(plan.keys()):
                raise ValueError(f"Missing keys: {required - plan.keys()}")

            print(f"[ai_director] ✓ Gemini plan: {plan}")
            return plan

        except json.JSONDecodeError as e:
            print(f"[ai_director] JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"[ai_director] Gemini error (attempt {attempt+1}): {e}")

        if attempt < retries:
            time.sleep(2)

    return None   # All attempts failed → caller uses fallback


# ── Step 2: Build a single clip segment using MoviePy ─────────────────────────

def _resolve_meme(meme_filename: str) -> Optional[Path]:
    """Find a meme insert file by name, case-insensitive. Returns None if missing."""
    target = meme_filename.strip().lower()
    for f in MEME_DIR.glob("*.mp4"):
        if f.name.lower() == target:
            return f
    # If exact match not found, pick any random meme
    available = list(MEME_DIR.glob("*.mp4"))
    if available:
        chosen = random.choice(available)
        print(f"[ai_director] Meme '{meme_filename}' not found — using {chosen.name}")
        return chosen
    return None


def _resize_video_for_canvas(clip: VideoFileClip) -> VideoFileClip:
    """
    Resize a VideoFileClip to fit within CARD_W (1080px) preserving aspect ratio.
    The clip is NOT cropped — it fits within the canvas.
    """
    src_w, src_h = clip.size
    scale = CARD_W / src_w
    new_w = CARD_W
    new_h = int(src_h * scale)
    # Cap height to avoid overflowing the canvas
    if new_h > CARD_H - 200:   # leave room for text/branding
        scale = (CARD_H - 200) / src_h
        new_w = int(src_w * scale)
        new_h = CARD_H - 200

    return clip.resize((new_w, new_h))


def _build_branding_clip(duration: float) -> TextClip:
    """Return a TextClip for the @madanimalx watermark at the bottom."""
    font_path = _find_font(50)
    try:
        brand = TextClip(
            BRAND_TEXT,
            fontsize=52,
            color="black",
            font=font_path if font_path else "Impact",
            method="caption",
            size=(CARD_W, None),
            align="center",
        ).set_duration(duration)
    except Exception:
        # Ultra-safe fallback if TextClip fails
        brand = ColorClip(size=(CARD_W, 60), color=[255, 255, 255]).set_duration(duration)
    return brand


def _build_punchline_clip(text: str, start: float, end: float) -> TextClip:
    """Return a TextClip for the Gemini-specified punchline at given time window."""
    font_path = _find_font(68)
    duration = end - start
    try:
        clip = TextClip(
            text,
            fontsize=68,
            color="black",
            font=font_path if font_path else "Impact",
            stroke_color="white",
            stroke_width=2,
            method="caption",
            size=(CARD_W - 80, None),
            align="center",
        ).set_start(start).set_duration(duration)
    except Exception as e:
        print(f"[ai_director] TextClip error: {e} — skipping punchline")
        clip = ColorClip(size=(0, 0), color=[255, 255, 255]).set_duration(0)
    return clip


def assemble_single_clip(
    raw_clip_path: Path,
    plan: Optional[dict],
    clip_index: int,
) -> Optional[Path]:
    """
    Assemble one processed Reel segment from a single raw clip + AI plan.

    If plan is None (fallback), assembles the clip without meme insert.

    Returns path to the processed segment .mp4, or None on failure.
    """
    dest = PROCESSED_DIR / f"segment_{clip_index:02d}.mp4"

    try:
        # ── Load and trim the source animal clip ──────────────────
        animal = VideoFileClip(str(raw_clip_path)).subclip(0, min(
            MAX_CLIP_DURATION,
            VideoFileClip(str(raw_clip_path)).duration
        ))

        total_duration = animal.duration  # may be split later

        # ── White card background ─────────────────────────────────
        white_bg = ImageClip(_make_white_card()).set_duration(total_duration)

        # ── Resize the animal video to fit the canvas ─────────────
        animal_resized = _resize_video_for_canvas(animal)
        av_w, av_h = animal_resized.size

        # Center the video on the card
        pos_x = (CARD_W - av_w) // 2
        pos_y = (CARD_H - av_h) // 2

        animal_positioned = animal_resized.set_position((pos_x, pos_y))

        # ── Branding (full duration) ───────────────────────────────
        brand_clip = _build_branding_clip(total_duration)
        brand_y = CARD_H - 100
        brand_clip = brand_clip.set_position(("center", brand_y))

        # ── Layers: background + video + brand ────────────────────
        layers = [white_bg, animal_positioned, brand_clip]

        # ── Punchline text (from AI plan) ─────────────────────────
        if plan:
            t_start = float(plan.get("text_start_time", 0.5))
            t_end = float(plan.get("text_end_time", 3.0))
            # Clamp to clip duration
            t_start = max(0.0, min(t_start, total_duration - 0.5))
            t_end = max(t_start + 0.5, min(t_end, total_duration))

            punchline = _build_punchline_clip(
                plan.get("punchline_text", ""),
                t_start, t_end
            ).set_position(("center", 60))
            layers.append(punchline)

        # ── Composite all layers into one clip ────────────────────
        composite = CompositeVideoClip(layers, size=(CARD_W, CARD_H))

        # ── Meme insert (split the video) ─────────────────────────
        if plan and plan.get("meme_insert_timestamp"):
            split_t = float(plan["meme_insert_timestamp"])
            split_t = max(1.0, min(split_t, total_duration - 1.0))

            meme_path = _resolve_meme(plan.get("meme_filename", ""))

            if meme_path and split_t < composite.duration:
                # Part 1: before the meme
                part1 = composite.subclip(0, split_t)

                # Meme clip on white card
                meme_raw = VideoFileClip(str(meme_path)).subclip(0, min(
                    MEME_DURATION, VideoFileClip(str(meme_path)).duration
                ))
                meme_resized = _resize_video_for_canvas(meme_raw)
                me_w, me_h = meme_resized.size
                meme_bg = ImageClip(_make_white_card()).set_duration(meme_raw.duration)
                meme_pos = meme_resized.set_position(
                    ((CARD_W - me_w) // 2, (CARD_H - me_h) // 2)
                )
                meme_brand = _build_branding_clip(meme_raw.duration).set_position(
                    ("center", brand_y)
                )
                meme_composite = CompositeVideoClip(
                    [meme_bg, meme_pos, meme_brand], size=(CARD_W, CARD_H)
                ).set_duration(meme_raw.duration)

                # Part 2: after the meme
                part2 = composite.subclip(split_t)

                composite = concatenate_videoclips([part1, meme_composite, part2])

        # ── Write segment to disk ─────────────────────────────────
        composite.write_videofile(
            str(dest),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
            logger=None,   # suppress verbose MoviePy output
        )
        composite.close()
        animal.close()
        print(f"[ai_director] ✓ Segment → {dest.name}")
        return dest

    except Exception as e:
        print(f"[ai_director] ERROR assembling {raw_clip_path.name}: {e}")
        return None


# ── Step 3: Assemble the full reel ────────────────────────────────────────────

def build_ai_reel(raw_clips: list[Path], gemini_key: str) -> Optional[Path]:
    """
    Main entrypoint. Processes each raw clip with Gemini analysis + MoviePy assembly.
    Falls back gracefully to no-meme-insert mode if Gemini fails.
    Returns path to final output .mp4.
    """
    segments: list[Path] = []

    for i, clip in enumerate(raw_clips):
        print(f"\n[ai_director] Processing clip {i+1}/{len(raw_clips)}: {clip.name}")

        # ── AI analysis ───────────────────────────────────────────
        if gemini_key:
            plan = analyse_with_gemini(clip, gemini_key)
            if not plan:
                print("[ai_director] Gemini failed — using standard fallback for this clip")
        else:
            print("[ai_director] No GEMINI_API_KEY — using fallback")
            plan = None

        # ── Assemble segment ──────────────────────────────────────
        seg = assemble_single_clip(clip, plan, i)
        if seg:
            segments.append(seg)

    if not segments:
        print("[ai_director] ERROR: No segments produced")
        return None

    # ── Concatenate all segments ──────────────────────────────────
    print(f"\n[ai_director] Concatenating {len(segments)} segments ...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"reel_ai_{timestamp}.mp4"

    try:
        all_clips = [VideoFileClip(str(s)) for s in segments]
        final = concatenate_videoclips(all_clips, method="compose")
        final.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
            logger=None,
        )
        final.close()
        for c in all_clips:
            c.close()

        size_mb = output_path.stat().st_size / 1_048_576
        print(f"\n[ai_director] ✅ Final reel: {output_path.name} ({size_mb:.1f} MB)")
        return output_path

    except Exception as e:
        print(f"[ai_director] ERROR during final concat: {e}")
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Director — Gemini + MoviePy Reel Assembly")
    parser.add_argument("--clips", nargs="+", required=True, help="Paths to raw .mp4 clips")
    parser.add_argument("--no-ai", action="store_true", help="Skip Gemini, use fallback mode")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(BASE_DIR.parent / ".env")

    gemini_key = "" if args.no_ai else os.getenv("GEMINI_API_KEY", "")
    clips = [Path(p) for p in args.clips if Path(p).exists()]

    if not clips:
        print("[ai_director] No valid clip paths provided")
        sys.exit(1)

    result = build_ai_reel(clips, gemini_key)
    if result:
        print(f"[ai_director] Output: {result}")
    else:
        sys.exit(1)

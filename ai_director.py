"""
ai_director.py — AI-Driven Video Editing Engine (v4)

v4 upgrades:
  - Tenor API: Fetches meme clips on-the-fly (no local meme folder required)
  - faster-whisper: Word-by-word dynamic captions (Hormozi style)
  - Gemini 1.5 Flash: Video analysis for punchline + meme selection
  - MoviePy: Assembly on White Meme Card template

Flow per clip:
  1. Gemini analysis → JSON plan (punchline, timestamps, meme keyword)
  2. Tenor API → downloads the best .mp4 for the meme keyword
  3. faster-whisper → word-level timestamps from audio
  4. MoviePy → white card + video + word captions + meme insert + branding
"""

import os
import sys
import json
import time
import random
import subprocess
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests as http_requests   # alias to avoid clash with google imports
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
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FONTS_DIR = BASE_DIR / "assets" / "fonts"
PROCESSED_DIR = BASE_DIR / ".tmp" / "processed"
MEME_CACHE_DIR = BASE_DIR / ".tmp" / "meme_cache"
OUTPUT_DIR = BASE_DIR / ".tmp" / "output"

for d in [PROCESSED_DIR, MEME_CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Canvas
CARD_W, CARD_H = 1080, 1920
BRAND_TEXT = "@madanimalx"
MAX_CLIP_DURATION = 10.0
MEME_DURATION = 3.0

# Font paths (fallback chain)
IMPACT_CANDIDATES = [
    FONTS_DIR / "Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# ── Gemini prompt ──────────────────────────────────────────────────────────────
DIRECTOR_PROMPT = """You are a viral Instagram Reels editor specializing in funny animal content.

Analyze the uploaded animal video carefully. Return ONLY a single valid JSON object — no markdown fences, no extra text.

Fields required:
{
  "punchline_text": "Short funny caption with 1-2 emojis (max 8 words)",
  "text_start_time": <float, seconds when text should appear>,
  "text_end_time": <float, seconds when text should disappear>,
  "meme_insert_timestamp": <float, exact second to pause video and insert meme>,
  "meme_search_query": "A 2-3 word Tenor search query for the perfect reaction meme (e.g. 'laughing cat', 'shocked pikachu', 'doge dance', 'facepalm')"
}

Rules:
- text timings must be within video duration
- meme_insert_timestamp must be > 1.0 and < (video_duration - 1.0)
- text_end_time must be > text_start_time
- punchline_text must be funny and match the action in the video
- meme_search_query must describe a reaction that fits the moment

Return ONLY the JSON object."""


# ══════════════════════════════════════════════════════════════════════════════
#  TENOR API: On-the-fly meme fetching
# ══════════════════════════════════════════════════════════════════════════════

TENOR_API_BASE = "https://tenor.googleapis.com/v2/search"

def fetch_tenor_meme(
    search_query: str,
    tenor_api_key: str = "",
) -> Optional[Path]:
    """
    Search Tenor for a meme GIF/MP4 by query. Downloads the top result as .mp4.

    Uses the free Tenor API v2 endpoint. If no API key is provided,
    tries with a basic anonymous request (may be rate-limited).

    Args:
        search_query: e.g. "laughing cat", "doge dance", "shocked pikachu"
        tenor_api_key: Google API key with Tenor API enabled (optional but recommended)

    Returns:
        Path to downloaded .mp4 meme clip, or None on failure.
    """
    # Sanitize query for caching
    cache_name = search_query.lower().replace(" ", "_")[:30]
    cached = MEME_CACHE_DIR / f"tenor_{cache_name}.mp4"
    if cached.exists() and cached.stat().st_size > 10_000:
        print(f"[ai_director] Tenor cache hit: {cached.name}")
        return cached

    # Build API request
    params = {
        "q": search_query,
        "limit": 5,
        "media_filter": "mp4",
        "contentfilter": "medium",
    }
    if tenor_api_key:
        params["key"] = tenor_api_key
    else:
        # Try anonymous — works but may be limited
        params["key"] = "AIzaSyAyimkuYQYF_FXVALexPuGQctUWRURdCYQ"  # Tenor public demo key

    try:
        print(f"[ai_director] Searching Tenor for: '{search_query}' ...")
        resp = http_requests.get(TENOR_API_BASE, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[ai_director] Tenor API error: HTTP {resp.status_code}")
            return None

        results = resp.json().get("results", [])
        if not results:
            print(f"[ai_director] Tenor: no results for '{search_query}'")
            return None

        # Find the best mp4 URL from the top result
        mp4_url = None
        for result in results:
            media_formats = result.get("media_formats", {})
            # Prefer mp4 format, then loopedmp4, then tinymp4
            for fmt in ["mp4", "loopedmp4", "tinymp4"]:
                if fmt in media_formats:
                    mp4_url = media_formats[fmt].get("url")
                    if mp4_url:
                        break
            if mp4_url:
                break

        if not mp4_url:
            print(f"[ai_director] Tenor: no mp4 format found for '{search_query}'")
            return None

        # Download the meme clip
        print(f"[ai_director] Downloading Tenor meme → {cached.name} ...")
        with http_requests.get(mp4_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(cached, "wb") as f:
                for chunk in r.iter_content(chunk_size=262144):
                    f.write(chunk)

        if cached.stat().st_size > 5_000:
            print(f"[ai_director] ✓ Tenor meme downloaded: {cached.name}")
            return cached
        else:
            cached.unlink()
            return None

    except Exception as e:
        print(f"[ai_director] Tenor error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  FASTER-WHISPER: Word-by-word caption extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio_wav(video_path: Path) -> Optional[Path]:
    """Extract audio from a video file as a 16kHz mono WAV."""
    wav_path = Path(tempfile.mktemp(suffix=".wav", dir="/tmp"))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        str(wav_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 1000:
            return wav_path
        print(f"[ai_director] Audio extraction failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"[ai_director] Audio extraction error: {e}")
    return None


def transcribe_with_whisper(audio_path: Path, model_size: str = "base") -> list[dict]:
    """
    Run faster-whisper on an audio file and return word-level timestamps.
    Returns: [{"word": "hello", "start": 0.5, "end": 0.8}, ...]
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ai_director] faster-whisper not installed — skipping dynamic captions")
        return []

    try:
        print(f"[ai_director] Loading faster-whisper '{model_size}' model ...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        print(f"[ai_director] Transcribing {audio_path.name} ...")
        segments, info = model.transcribe(
            str(audio_path), beam_size=5, word_timestamps=True, language=None,
        )

        words = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                    })

        print(f"[ai_director] ✓ Transcribed {len(words)} words")
        return words
    except Exception as e:
        print(f"[ai_director] Whisper error: {e}")
        return []


def build_word_caption_clips(
    words: list[dict],
    video_y_top: int,
    video_height: int,
) -> list:
    """
    Create MoviePy TextClip for each transcribed word.
    Hormozi style: yellow text, thick black stroke, centered, timed per-word.
    Positioned in the lower-third of the video area.
    """
    if not words:
        return []

    font_path = _find_font(80)
    clips = []
    caption_y = video_y_top + int(video_height * 0.75)

    for w in words:
        word_text = w["word"].upper()
        duration = w["end"] - w["start"]
        if duration < 0.05 or not word_text.strip():
            continue
        try:
            clip = TextClip(
                word_text,
                fontsize=80,
                color="yellow",
                font=font_path if font_path else "Impact",
                stroke_color="black",
                stroke_width=4,
                method="caption",
                size=(CARD_W - 100, None),
                align="center",
            ).set_start(w["start"]).set_duration(duration).set_position(
                ("center", caption_y)
            )
            clips.append(clip)
        except Exception:
            continue

    print(f"[ai_director] ✓ Built {len(clips)} word caption overlays")
    return clips


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI: Video analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_with_gemini(
    video_path: Path,
    gemini_key: str,
    retries: int = 2,
) -> Optional[dict]:
    """Upload video to Gemini 1.5 Flash, return JSON editing plan."""
    client = genai.Client(api_key=gemini_key)

    for attempt in range(retries + 1):
        try:
            print(f"[ai_director] Uploading {video_path.name} to Gemini ...")
            uploaded = client.files.upload(
                path=str(video_path),
                config={"mime_type": "video/mp4", "display_name": video_path.stem},
            )

            for _ in range(20):
                file_info = client.files.get(name=uploaded.name)
                if file_info.state.name == "ACTIVE":
                    break
                print("[ai_director] Waiting for file processing ...")
                time.sleep(3)
            else:
                return None

            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[uploaded, DIRECTOR_PROMPT],
                config=genai_types.GenerateContentConfig(
                    temperature=0.7, max_output_tokens=500,
                ),
            )

            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

            raw = response.text.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            plan = json.loads(raw)
            required = {"punchline_text", "text_start_time", "text_end_time",
                        "meme_insert_timestamp", "meme_search_query"}
            if not required.issubset(plan.keys()):
                raise ValueError(f"Missing keys: {required - plan.keys()}")

            print(f"[ai_director] ✓ Gemini plan: {json.dumps(plan, indent=2)}")
            return plan

        except json.JSONDecodeError as e:
            print(f"[ai_director] JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"[ai_director] Gemini error (attempt {attempt+1}): {e}")
        if attempt < retries:
            time.sleep(2)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MOVIEPY: Assembly helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_font(size: int = 60) -> str:
    for p in IMPACT_CANDIDATES:
        if Path(p).exists():
            return str(p)
    return ""


def _make_white_card() -> np.ndarray:
    return np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)


def _resize_video_for_canvas(clip: VideoFileClip) -> tuple:
    """Resize to fit 1080px wide, preserving AR. Returns (clip, w, h)."""
    src_w, src_h = clip.size
    scale = CARD_W / src_w
    new_w = CARD_W
    new_h = int(src_h * scale)
    if new_h > CARD_H - 200:
        scale = (CARD_H - 200) / src_h
        new_w = int(src_w * scale)
        new_h = CARD_H - 200
    return clip.resize((new_w, new_h)), new_w, new_h


def _build_branding_clip(duration: float) -> TextClip:
    font_path = _find_font(50)
    try:
        return TextClip(
            BRAND_TEXT, fontsize=52, color="black",
            font=font_path if font_path else "Impact",
            method="caption", size=(CARD_W, None), align="center",
        ).set_duration(duration)
    except Exception:
        return ColorClip(size=(CARD_W, 60), color=[255, 255, 255]).set_duration(duration)


def _build_punchline_clip(text: str, start: float, end: float) -> TextClip:
    font_path = _find_font(68)
    duration = end - start
    try:
        return TextClip(
            text, fontsize=68, color="black",
            font=font_path if font_path else "Impact",
            stroke_color="white", stroke_width=2,
            method="caption", size=(CARD_W - 80, None), align="center",
        ).set_start(start).set_duration(duration)
    except Exception as e:
        print(f"[ai_director] TextClip error: {e}")
        return ColorClip(size=(0, 0), color=[255, 255, 255]).set_duration(0)


# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLY: Build a single clip segment
# ══════════════════════════════════════════════════════════════════════════════

def assemble_single_clip(
    raw_clip_path: Path,
    plan: Optional[dict],
    clip_index: int,
    tenor_api_key: str = "",
) -> Optional[Path]:
    """
    Assemble one Reel segment:
      1. White card canvas + centered video (natural AR)
      2. faster-whisper word-by-word captions (yellow/black, Hormozi style)
      3. Gemini punchline text at top
      4. Tenor meme insert at Gemini-specified timestamp
      5. @madanimalx branding at bottom
    """
    dest = PROCESSED_DIR / f"segment_{clip_index:02d}.mp4"

    try:
        # ── Load and trim ─────────────────────────────────────────
        animal = VideoFileClip(str(raw_clip_path)).subclip(0, min(
            MAX_CLIP_DURATION,
            VideoFileClip(str(raw_clip_path)).duration
        ))
        total_duration = animal.duration

        # ── faster-whisper: extract word captions ─────────────────
        wav_path = extract_audio_wav(raw_clip_path)
        words = transcribe_with_whisper(wav_path) if wav_path else []
        if wav_path:
            try:
                wav_path.unlink()
            except Exception:
                pass

        # ── White card background ─────────────────────────────────
        white_bg = ImageClip(_make_white_card()).set_duration(total_duration)

        # ── Resize video + center on card ─────────────────────────
        animal_resized, av_w, av_h = _resize_video_for_canvas(animal)
        pos_x = (CARD_W - av_w) // 2
        pos_y = (CARD_H - av_h) // 2
        animal_positioned = animal_resized.set_position((pos_x, pos_y))

        # ── Word-by-word dynamic captions ─────────────────────────
        word_clips = build_word_caption_clips(words, pos_y, av_h) if words else []

        # ── Branding ──────────────────────────────────────────────
        brand_clip = _build_branding_clip(total_duration)
        brand_y = CARD_H - 100
        brand_clip = brand_clip.set_position(("center", brand_y))

        # ── Compose layers ────────────────────────────────────────
        layers = [white_bg, animal_positioned] + word_clips + [brand_clip]

        # ── Gemini punchline ──────────────────────────────────────
        if plan:
            t_start = max(0.0, min(float(plan.get("text_start_time", 0.5)), total_duration - 0.5))
            t_end = max(t_start + 0.5, min(float(plan.get("text_end_time", 3.0)), total_duration))
            punchline = _build_punchline_clip(
                plan.get("punchline_text", ""), t_start, t_end
            ).set_position(("center", 60))
            layers.append(punchline)

        composite = CompositeVideoClip(layers, size=(CARD_W, CARD_H))

        # ── Tenor meme insert (on-the-fly download) ───────────────
        if plan and plan.get("meme_insert_timestamp") and plan.get("meme_search_query"):
            split_t = float(plan["meme_insert_timestamp"])
            split_t = max(1.0, min(split_t, total_duration - 1.0))

            # Fetch meme from Tenor API
            meme_path = fetch_tenor_meme(
                plan["meme_search_query"],
                tenor_api_key=tenor_api_key,
            )

            if meme_path and split_t < composite.duration:
                part1 = composite.subclip(0, split_t)

                meme_raw = VideoFileClip(str(meme_path)).subclip(0, min(
                    MEME_DURATION, VideoFileClip(str(meme_path)).duration
                ))
                meme_resized, me_w, me_h = _resize_video_for_canvas(meme_raw)
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

                part2 = composite.subclip(split_t)
                composite = concatenate_videoclips([part1, meme_composite, part2])

        # ── Write segment ─────────────────────────────────────────
        composite.write_videofile(
            str(dest), fps=30, codec="libx264", audio_codec="aac",
            preset="fast", ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
            logger=None,
        )
        composite.close()
        animal.close()
        print(f"[ai_director] ✓ Segment → {dest.name}")
        return dest

    except Exception as e:
        print(f"[ai_director] ERROR assembling {raw_clip_path.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  FULL REEL: Process all clips and concatenate
# ══════════════════════════════════════════════════════════════════════════════

def build_ai_reel(raw_clips: list[Path], gemini_key: str, tenor_key: str = "") -> Optional[Path]:
    """Main entrypoint. Process each clip with Gemini + Whisper + Tenor + MoviePy."""
    segments: list[Path] = []

    for i, clip in enumerate(raw_clips):
        print(f"\n[ai_director] Processing clip {i+1}/{len(raw_clips)}: {clip.name}")

        plan = None
        if gemini_key:
            plan = analyse_with_gemini(clip, gemini_key)
            if not plan:
                print("[ai_director] Gemini failed — using fallback")
        else:
            print("[ai_director] No GEMINI_API_KEY — fallback mode")

        seg = assemble_single_clip(clip, plan, i, tenor_api_key=tenor_key)
        if seg:
            segments.append(seg)

    if not segments:
        print("[ai_director] ERROR: No segments produced")
        return None

    print(f"\n[ai_director] Concatenating {len(segments)} segments ...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"reel_ai_{timestamp}.mp4"

    try:
        all_clips = [VideoFileClip(str(s)) for s in segments]
        final = concatenate_videoclips(all_clips, method="compose")
        final.write_videofile(
            str(output_path), fps=30, codec="libx264", audio_codec="aac",
            preset="fast", ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
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
    parser = argparse.ArgumentParser(description="AI Director v4 — Gemini + Whisper + Tenor")
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--no-ai", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(BASE_DIR.parent / ".env")

    gemini_key = "" if args.no_ai else os.getenv("GEMINI_API_KEY", "")
    tenor_key = os.getenv("TENOR_API_KEY", "")
    clips = [Path(p) for p in args.clips if Path(p).exists()]

    if not clips:
        print("[ai_director] No valid clip paths")
        sys.exit(1)

    result = build_ai_reel(clips, gemini_key, tenor_key)
    sys.exit(0 if result else 1)

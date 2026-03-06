"""
ai_director.py — AI-Driven Video Editing Engine (v3)

v3 upgrades:
  - faster-whisper: Extracts word-level timestamps from audio → renders
    dynamic word-by-word captions on screen (Alex Hormozi caption style)
  - Gemini 1.5 Flash: Analyzes each clip for punchline text + meme inserts
  - MoviePy: Assembles everything on a White Meme Card template

Caption rendering flow:
  1. Extract audio from raw clip → /tmp WAV
  2. Run faster-whisper (base model) → get segments with word-level timestamps
  3. Create a MoviePy TextClip per word, timed precisely to when it's spoken
  4. Overlay all word clips over the center video on the white card

No TTS or AI voice generation — captions are pure visual text overlays.
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


# ══════════════════════════════════════════════════════════════════════════════
#  FASTER-WHISPER: Word-by-word caption extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio_wav(video_path: Path) -> Optional[Path]:
    """
    Extract audio from a video file as a 16kHz mono WAV.
    faster-whisper works best with 16kHz mono input.
    Returns path to the temporary WAV file, or None on failure.
    """
    wav_path = Path(tempfile.mktemp(suffix=".wav", dir="/tmp"))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1",             # mono
        "-ar", "16000",         # 16kHz
        "-vn",                  # no video
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

    Returns a list of dicts:
        [{"word": "hello", "start": 0.5, "end": 0.8}, ...]

    Uses the 'base' model by default for speed. Switch to 'small' for
    better accuracy on noisy animal videos.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ai_director] faster-whisper not installed — skipping dynamic captions")
        return []

    try:
        print(f"[ai_director] Loading faster-whisper '{model_size}' model ...")
        model = WhisperModel(
            model_size,
            device="cpu",          # Use GPU if available: device="cuda"
            compute_type="int8",   # Fast inference on CPU
        )

        print(f"[ai_director] Transcribing {audio_path.name} ...")
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=True,  # CRITICAL: enables per-word timing
            language=None,         # auto-detect language
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
        print(f"[ai_director] Whisper transcription error: {e}")
        return []


def build_word_caption_clips(
    words: list[dict],
    video_y_top: int,
    video_height: int,
) -> list:
    """
    Create MoviePy TextClip objects for each transcribed word, rendered
    word-by-word over the video area (Alex Hormozi caption style).

    Each word appears at its exact spoken timestamp and disappears when done.
    Positioned in the lower-third of the video area for readability.

    Args:
        words: list of {"word", "start", "end"} dicts from faster-whisper
        video_y_top: y-coordinate where the video starts on the white card
        video_height: height of the video area

    Returns:
        List of positioned, timed TextClip objects ready for CompositeVideoClip.
    """
    if not words:
        return []

    font_path = _find_font(80)
    clips = []

    # Caption y-position: lower third of the video area
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
                stroke_width=4,       # thick black stroke for visibility
                method="caption",
                size=(CARD_W - 100, None),
                align="center",
            ).set_start(w["start"]).set_duration(duration).set_position(
                ("center", caption_y)
            )
            clips.append(clip)
        except Exception as e:
            # Skip words that fail to render (emoji, special chars)
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
    """
    Upload video to Gemini 1.5 Flash and return JSON editing plan.
    Returns None on failure (triggers fallback).
    """
    client = genai.Client(api_key=gemini_key)

    for attempt in range(retries + 1):
        try:
            print(f"[ai_director] Uploading {video_path.name} to Gemini Files API ...")
            uploaded = client.files.upload(
                path=str(video_path),
                config={"mime_type": "video/mp4", "display_name": video_path.stem},
            )

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

            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

            raw_text = response.text.strip()
            raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            plan = json.loads(raw_text)
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

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MOVIEPY: Assembly helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_font(size: int = 60) -> str:
    """Return path string to the best available Impact-style font."""
    for p in IMPACT_CANDIDATES:
        if Path(p).exists():
            return str(p)
    return ""


def _make_white_card() -> np.ndarray:
    """Return a numpy array (H×W×3) of a pure white 1080x1920 frame."""
    return np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)


def _resolve_meme(meme_filename: str) -> Optional[Path]:
    """Find a meme insert file by name, case-insensitive."""
    target = meme_filename.strip().lower()
    for f in MEME_DIR.glob("*.mp4"):
        if f.name.lower() == target:
            return f
    available = list(MEME_DIR.glob("*.mp4"))
    if available:
        chosen = random.choice(available)
        print(f"[ai_director] Meme '{meme_filename}' not found — using {chosen.name}")
        return chosen
    return None


def _resize_video_for_canvas(clip: VideoFileClip) -> tuple:
    """
    Resize a VideoFileClip to fit within CARD_W (1080px) preserving aspect ratio.
    Returns (resized_clip, new_w, new_h).
    """
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
        brand = ColorClip(size=(CARD_W, 60), color=[255, 255, 255]).set_duration(duration)
    return brand


def _build_punchline_clip(text: str, start: float, end: float) -> TextClip:
    """Return a TextClip for the Gemini-specified punchline."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLY: Build a single clip segment
# ══════════════════════════════════════════════════════════════════════════════

def assemble_single_clip(
    raw_clip_path: Path,
    plan: Optional[dict],
    clip_index: int,
) -> Optional[Path]:
    """
    Assemble one processed Reel segment from a raw clip + AI plan + whisper captions.

    Pipeline:
      1. Load & trim animal clip
      2. faster-whisper → word-level timestamps
      3. White card canvas + centered video
      4. Word-by-word dynamic captions (yellow text, black stroke) over video
      5. Gemini punchline text (if available)
      6. @madanimalx branding
      7. Meme insert split (if Gemini specified)
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
        word_clips = []
        wav_path = extract_audio_wav(raw_clip_path)
        if wav_path:
            words = transcribe_with_whisper(wav_path, model_size="base")
            # Clean up temp WAV
            try:
                wav_path.unlink()
            except Exception:
                pass
        else:
            words = []

        # ── White card background ─────────────────────────────────
        white_bg = ImageClip(_make_white_card()).set_duration(total_duration)

        # ── Resize video and position on card ─────────────────────
        animal_resized, av_w, av_h = _resize_video_for_canvas(animal)
        pos_x = (CARD_W - av_w) // 2
        pos_y = (CARD_H - av_h) // 2
        animal_positioned = animal_resized.set_position((pos_x, pos_y))

        # ── Build word-by-word caption clips ──────────────────────
        # Position captions over the lower-third of the video area
        if words:
            word_clips = build_word_caption_clips(
                words,
                video_y_top=pos_y,
                video_height=av_h,
            )

        # ── Branding at bottom ────────────────────────────────────
        brand_clip = _build_branding_clip(total_duration)
        brand_y = CARD_H - 100
        brand_clip = brand_clip.set_position(("center", brand_y))

        # ── Layers: background + video + word captions + brand ────
        layers = [white_bg, animal_positioned] + word_clips + [brand_clip]

        # ── Gemini punchline text (top of card) ───────────────────
        if plan:
            t_start = float(plan.get("text_start_time", 0.5))
            t_end = float(plan.get("text_end_time", 3.0))
            t_start = max(0.0, min(t_start, total_duration - 0.5))
            t_end = max(t_start + 0.5, min(t_end, total_duration))

            punchline = _build_punchline_clip(
                plan.get("punchline_text", ""),
                t_start, t_end
            ).set_position(("center", 60))
            layers.append(punchline)

        # ── Composite ─────────────────────────────────────────────
        composite = CompositeVideoClip(layers, size=(CARD_W, CARD_H))

        # ── Meme insert (split the video at Gemini timestamp) ─────
        if plan and plan.get("meme_insert_timestamp"):
            split_t = float(plan["meme_insert_timestamp"])
            split_t = max(1.0, min(split_t, total_duration - 1.0))
            meme_path = _resolve_meme(plan.get("meme_filename", ""))

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
            str(dest),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
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

def build_ai_reel(raw_clips: list[Path], gemini_key: str) -> Optional[Path]:
    """
    Main entrypoint. For each raw clip:
      1. Gemini analysis (punchline + meme timing)
      2. faster-whisper transcription (word-level captions)
      3. MoviePy assembly (white card + video + captions + brand)
    Falls back gracefully on any individual failure.
    """
    segments: list[Path] = []

    for i, clip in enumerate(raw_clips):
        print(f"\n[ai_director] Processing clip {i+1}/{len(raw_clips)}: {clip.name}")

        # ── AI analysis ───────────────────────────────────────────
        plan = None
        if gemini_key:
            plan = analyse_with_gemini(clip, gemini_key)
            if not plan:
                print("[ai_director] Gemini failed — using fallback for this clip")
        else:
            print("[ai_director] No GEMINI_API_KEY — using fallback")

        # ── Assemble segment (includes whisper captions) ──────────
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
    parser = argparse.ArgumentParser(description="AI Director v3 — Gemini + Whisper + MoviePy")
    parser.add_argument("--clips", nargs="+", required=True, help="Paths to raw .mp4 clips")
    parser.add_argument("--no-ai", action="store_true", help="Skip Gemini, use fallback mode")
    parser.add_argument("--whisper-model", default="base", help="Whisper model: tiny/base/small")
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

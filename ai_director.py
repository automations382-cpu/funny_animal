"""
ai_director.py — Dual-AI Video Editing Engine (v5 — Groq + OpenAI)

AI Stack:
  • GPT-4o (via GitHub Models endpoint) — Analyses faster-whisper transcript,
    decides punchline text, timing, and meme keyword.
  • Groq (Llama 3) — Generates Instagram caption + hashtags from the video title.
  • Giphy API — Downloads meme clips on the fly (no local folder).
  • faster-whisper — Word-level timestamps for Hormozi-style captions.
  • MoviePy — Composites everything onto the White Meme Card.

NO Gemini. NO local meme_inserts folder.
"""

import os, sys, json, time, random, subprocess, tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests as http_req
import numpy as np
from moviepy.editor import (
    VideoFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips, TextClip, ColorClip,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FONTS_DIR = BASE_DIR / "assets" / "fonts"
PROCESSED_DIR = BASE_DIR / ".tmp" / "processed"
MEME_CACHE = BASE_DIR / ".tmp" / "meme_cache"
OUTPUT_DIR = BASE_DIR / ".tmp" / "output"
CAPTIONS_DIR = BASE_DIR / ".tmp" / "captions"

for d in [PROCESSED_DIR, MEME_CACHE, OUTPUT_DIR, CAPTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CARD_W, CARD_H = 1080, 1920
BRAND = "@madanimalx"
MAX_CLIP = 10.0
MEME_DUR = 3.0

FONT_CANDIDATES = [
    FONTS_DIR / "Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# ── GPT-4o prompt (editing decisions) ──────────────────────────────────────────
GPT4O_PROMPT = """You are a viral Instagram Reels editor for funny animal content.

I will give you a word-by-word transcript with timestamps from an animal video.
Analyze it and return ONLY a valid JSON object — no markdown fences, no extra text.

{
  "punchline_text": "Short funny caption with 1-2 emojis (max 8 words)",
  "text_start_time": <float seconds>,
  "text_end_time": <float seconds>,
  "meme_insert_timestamp": <float, exact second to pause video and insert a meme>,
  "meme_search_query": "2-3 word Giphy search query for a reaction meme (e.g. laughing cat, shocked pikachu)"
}

Rules:
- All times must be within 0 and the video duration
- meme_insert_timestamp > 1.0 and < (duration - 1.0)
- text_end_time > text_start_time
- punchline_text must be funny and relevant
Return ONLY the JSON."""


# ══════════════════════════════════════════════════════════════════════════════
#  GPT-4o via GitHub Models (free endpoint)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_with_gpt4o(
    transcript_text: str,
    video_duration: float,
    github_token: str,
) -> Optional[dict]:
    """
    Send the faster-whisper transcript to GPT-4o via the free GitHub Models
    endpoint to get editing decisions (punchline, timestamps, meme keyword).
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[ai_director] openai package not installed")
        return None

    if not github_token:
        print("[ai_director] No GITHUB_TOKEN — skipping GPT-4o analysis")
        return None

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token,
    )

    user_msg = (
        f"Video duration: {video_duration:.1f} seconds.\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Return the JSON editing plan."
    )

    for attempt in range(3):
        try:
            print(f"[ai_director] Sending transcript to GPT-4o (attempt {attempt+1}) ...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": GPT4O_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            plan = json.loads(raw)
            required = {"punchline_text", "text_start_time", "text_end_time",
                        "meme_insert_timestamp", "meme_search_query"}
            if not required.issubset(plan.keys()):
                raise ValueError(f"Missing keys: {required - plan.keys()}")

            print(f"[ai_director] ✓ GPT-4o plan: {json.dumps(plan)}")
            return plan

        except json.JSONDecodeError as e:
            print(f"[ai_director] JSON parse error: {e}")
        except Exception as e:
            print(f"[ai_director] GPT-4o error: {e}")
        time.sleep(1)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Groq (Llama 3) — Instagram caption generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_caption_groq(
    video_title: str,
    groq_api_key: str,
) -> str:
    """
    Use Groq's Llama 3 to generate an Instagram caption with emojis and hashtags
    from the video title. Saves to a .txt file and returns the caption text.
    """
    try:
        from groq import Groq
    except ImportError:
        print("[ai_director] groq package not installed")
        return _fallback_caption(video_title)

    if not groq_api_key:
        print("[ai_director] No GROQ_API_KEY — using fallback caption")
        return _fallback_caption(video_title)

    client = Groq(api_key=groq_api_key)

    prompt = (
        "You are a viral Instagram content creator for a funny animal memes page called @madanimalx.\n\n"
        f"Video title/context: \"{video_title}\"\n\n"
        "Write a short, highly engaging Instagram caption (2-3 sentences max) with relevant emojis. "
        "Then add a line break and 15-20 viral hashtags on a separate line.\n"
        "Make it funny, relatable, and shareable. Do NOT use markdown formatting."
    )

    try:
        print(f"[ai_director] Generating caption via Groq (Llama 3) ...")
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You write viral Instagram captions for animal content."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=300,
        )
        caption = response.choices[0].message.content.strip()
        print(f"[ai_director] ✓ Groq caption ({len(caption)} chars)")
        return caption
    except Exception as e:
        print(f"[ai_director] Groq error: {e}")
        return _fallback_caption(video_title)


def _fallback_caption(title: str) -> str:
    options = [
        f"When animals choose violence 😂💀\n\n{title}\n\n"
        "#funnyanimals #memes #viral #comedy #pets #catlover #doglover "
        "#animalmemes #reels #trending #explore #lol #cute #madanimalx",
        f"This one got me 😭🤣\n\n{title}\n\n"
        "#animals #funny #pet #meme #viral #reels #trending #comedy "
        "#cute #lmao #doglover #catlover #madanimalx #explore #fyp",
    ]
    return random.choice(options)


def save_caption(caption: str, video_path: Path) -> Path:
    """Save caption to a .txt file alongside the video."""
    txt_path = video_path.with_suffix(".txt")
    txt_path.write_text(caption, encoding="utf-8")
    print(f"[ai_director] ✓ Caption saved → {txt_path.name}")
    return txt_path


# ══════════════════════════════════════════════════════════════════════════════
#  Giphy API — On-the-fly meme fetching
# ══════════════════════════════════════════════════════════════════════════════

GIPHY_API = "https://api.giphy.com/v1/gifs/search"

def fetch_giphy_meme(query: str, giphy_key: str = "") -> Optional[Path]:
    """Search Giphy for a meme .mp4 by keyword. Caches results."""
    name = query.lower().replace(" ", "_").replace("/", "")[:30]
    cached = MEME_CACHE / f"giphy_{name}.mp4"
    if cached.exists() and cached.stat().st_size > 5_000:
        print(f"[ai_director] Giphy cache hit: {cached.name}")
        return cached

    if not giphy_key:
        print("[ai_director] No GIPHY_API_KEY provided")
        return None

    params = {
        "api_key": giphy_key,
        "q": query,
        "limit": 1
    }
    
    try:
        print(f"[ai_director] Giphy search: '{query}' ...")
        resp = http_req.get(GIPHY_API, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[ai_director] Giphy HTTP {resp.status_code}: {resp.text}")
            return None
            
        data = resp.json().get("data", [])
        if not data:
            print(f"[ai_director] Giphy returned no results for '{query}'")
            return None
            
        mp4_url = data[0].get("images", {}).get("original", {}).get("mp4")
        if not mp4_url:
            print(f"[ai_director] Giphy result missing MP4 URL")
            return None

        with http_req.get(mp4_url, stream=True, timeout=30) as dl:
            dl.raise_for_status()
            with open(cached, "wb") as f:
                for chunk in dl.iter_content(262144):
                    f.write(chunk)
                    
        if cached.stat().st_size > 5_000:
            print(f"[ai_director] ✓ Giphy meme → {cached.name}")
            return cached
            
        cached.unlink()
    except Exception as e:
        print(f"[ai_director] Giphy error: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  faster-whisper: Word-level transcription
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio(video: Path) -> Optional[Path]:
    wav = Path(tempfile.mktemp(suffix=".wav", dir="/tmp"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
           "-ac", "1", "-ar", "16000", "-vn", str(wav)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and wav.exists() and wav.stat().st_size > 1000:
            return wav
    except Exception as e:
        print(f"[ai_director] Audio extract error: {e}")
    return None


def transcribe(audio: Path, model_size: str = "base") -> list[dict]:
    """Run faster-whisper → list of {word, start, end}."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ai_director] faster-whisper not installed")
        return []
    try:
        print(f"[ai_director] Transcribing with faster-whisper ({model_size}) ...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio), beam_size=5, word_timestamps=True)
        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word.strip(), "start": round(w.start, 3),
                                  "end": round(w.end, 3)})
        print(f"[ai_director] ✓ {len(words)} words transcribed")
        return words
    except Exception as e:
        print(f"[ai_director] Whisper error: {e}")
        return []


def words_to_transcript_text(words: list[dict], duration: float) -> str:
    """Format words into a readable transcript for GPT-4o."""
    if not words:
        return f"[No speech detected in this {duration:.1f}s animal video — just animal sounds and action]"
    lines = []
    for w in words:
        lines.append(f"[{w['start']:.1f}s-{w['end']:.1f}s] {w['word']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MoviePy: Rendering helpers
# ══════════════════════════════════════════════════════════════════════════════

def _font(size: int = 60) -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return str(p)
    return ""


def _white_frame() -> np.ndarray:
    return np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)


def _resize_for_card(clip: VideoFileClip):
    sw, sh = clip.size
    scale = CARD_W / sw
    nw, nh = CARD_W, int(sh * scale)
    if nh > CARD_H - 200:
        scale = (CARD_H - 200) / sh
        nw, nh = int(sw * scale), CARD_H - 200
    return clip.resize((nw, nh)), nw, nh


def _brand(dur: float):
    f = _font(50)
    try:
        return TextClip(BRAND, fontsize=52, color="black",
                        font=f if f else "Impact",
                        method="caption", size=(CARD_W, None),
                        align="center").set_duration(dur)
    except Exception:
        return ColorClip(size=(CARD_W, 60), color=[255, 255, 255]).set_duration(dur)


def word_caption_clips(words: list[dict], vid_y: int, vid_h: int) -> list:
    """Hormozi-style word-by-word captions: yellow text, black stroke."""
    if not words:
        return []
    f = _font(80)
    clips = []
    cy = vid_y + int(vid_h * 0.75)
    for w in words:
        txt = w["word"].upper()
        dur = w["end"] - w["start"]
        if dur < 0.05 or not txt.strip():
            continue
        try:
            c = TextClip(txt, fontsize=80, color="yellow",
                         font=f if f else "Impact",
                         stroke_color="black", stroke_width=4,
                         method="caption", size=(CARD_W - 100, None),
                         align="center"
                         ).set_start(w["start"]).set_duration(dur).set_position(("center", cy))
            clips.append(c)
        except Exception:
            continue
    print(f"[ai_director] ✓ {len(clips)} word overlays built")
    return clips


def _punchline(text: str, start: float, end: float):
    f = _font(68)
    try:
        return TextClip(text, fontsize=68, color="black",
                        font=f if f else "Impact",
                        stroke_color="white", stroke_width=2,
                        method="caption", size=(CARD_W - 80, None),
                        align="center"
                        ).set_start(start).set_duration(end - start)
    except Exception:
        return ColorClip(size=(0, 0), color=[255, 255, 255]).set_duration(0)


# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLY: Single clip → segment
# ══════════════════════════════════════════════════════════════════════════════

def assemble_segment(
    raw_clip: Path,
    plan: Optional[dict],
    words: list[dict],
    idx: int,
    giphy_key: str = "",
) -> Optional[Path]:
    """
    Build one processed segment:
      White card + centered video + word captions + punchline + meme insert + brand.
    """
    dest = PROCESSED_DIR / f"segment_{idx:02d}.mp4"
    try:
        animal = VideoFileClip(str(raw_clip)).subclip(0, min(
            MAX_CLIP, VideoFileClip(str(raw_clip)).duration))
        dur = animal.duration

        bg = ImageClip(_white_frame()).set_duration(dur)
        vid, vw, vh = _resize_for_card(animal)
        px, py = (CARD_W - vw) // 2, (CARD_H - vh) // 2
        vid = vid.set_position((px, py))

        wcaps = word_caption_clips(words, py, vh)
        brand = _brand(dur).set_position(("center", CARD_H - 100))

        layers = [bg, vid] + wcaps + [brand]

        if plan:
            ts = max(0, min(float(plan.get("text_start_time", 0.5)), dur - 0.5))
            te = max(ts + 0.5, min(float(plan.get("text_end_time", 3)), dur))
            layers.append(_punchline(plan.get("punchline_text", ""), ts, te)
                          .set_position(("center", 60)))

        comp = CompositeVideoClip(layers, size=(CARD_W, CARD_H))

        # Meme insert
        if plan and plan.get("meme_insert_timestamp") and plan.get("meme_search_query"):
            split = max(1.0, min(float(plan["meme_insert_timestamp"]), dur - 1.0))
            meme = fetch_giphy_meme(plan["meme_search_query"], giphy_key)
            if meme and split < comp.duration:
                p1 = comp.subclip(0, split)
                mr = VideoFileClip(str(meme)).subclip(0, min(
                    MEME_DUR, VideoFileClip(str(meme)).duration))
                mrr, mw, mh = _resize_for_card(mr)
                mbg = ImageClip(_white_frame()).set_duration(mr.duration)
                mc = CompositeVideoClip([
                    mbg, mrr.set_position(((CARD_W - mw) // 2, (CARD_H - mh) // 2)),
                    _brand(mr.duration).set_position(("center", CARD_H - 100))
                ], size=(CARD_W, CARD_H)).set_duration(mr.duration)
                p2 = comp.subclip(split)
                comp = concatenate_videoclips([p1, mc, p2])

        comp.write_videofile(str(dest), fps=30, codec="libx264", audio_codec="aac",
                             preset="fast", ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
                             logger=None)
        comp.close()
        animal.close()
        print(f"[ai_director] ✓ Segment → {dest.name}")
        return dest
    except Exception as e:
        print(f"[ai_director] ERROR {raw_clip.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  FULL REEL: Process all clips
# ══════════════════════════════════════════════════════════════════════════════

def build_ai_reel(
    raw_clips: list[Path],
    github_token: str = "",
    groq_key: str = "",
    giphy_key: str = "",
    video_titles: Optional[list[str]] = None,
) -> Optional[Path]:
    """
    Main entrypoint.
    For each clip:
      1. faster-whisper → word-level transcript
      2. GPT-4o (GitHub) → editing plan (punchline, timestamps, meme query)
      3. Groq (Llama 3) → Instagram caption
      4. Giphy → download meme on the fly
      5. MoviePy → white card assembly
    """
    segments: list[Path] = []
    all_captions: list[str] = []
    titles = video_titles or ["Funny animal clip"] * len(raw_clips)

    for i, clip in enumerate(raw_clips):
        print(f"\n{'='*50}")
        print(f"[ai_director] Clip {i+1}/{len(raw_clips)}: {clip.name}")
        print(f"{'='*50}")

        # ── Whisper transcription ─────────────────────────────────
        wav = extract_audio(clip)
        words = transcribe(wav) if wav else []
        if wav:
            try: wav.unlink()
            except Exception: pass

        # ── GPT-4o editing plan ───────────────────────────────────
        vdur = VideoFileClip(str(clip)).duration
        transcript_txt = words_to_transcript_text(words, vdur)
        plan = analyse_with_gpt4o(transcript_txt, vdur, github_token)
        if not plan:
            print("[ai_director] GPT-4o unavailable — fallback assembly")

        # ── Groq caption ─────────────────────────────────────────
        title = titles[i] if i < len(titles) else "Funny animal clip"
        caption = generate_caption_groq(title, groq_key)
        all_captions.append(caption)

        # ── Assemble ──────────────────────────────────────────────
        seg = assemble_segment(clip, plan, words, i, giphy_key)
        if seg:
            segments.append(seg)

    if not segments:
        print("[ai_director] ERROR: No segments produced")
        return None

    # ── Concatenate ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"reel_ai_{ts}.mp4"
    print(f"\n[ai_director] Concatenating {len(segments)} segments ...")
    try:
        clips = [VideoFileClip(str(s)) for s in segments]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(str(out), fps=30, codec="libx264", audio_codec="aac",
                              preset="fast", ffmpeg_params=["-crf", "23", "-movflags", "+faststart"],
                              logger=None)
        final.close()
        for c in clips: c.close()
    except Exception as e:
        print(f"[ai_director] Concat error: {e}")
        return None

    # Save caption for the best/first one (used by uploader)
    if all_captions:
        save_caption(all_captions[0], out)

    mb = out.stat().st_size / 1_048_576
    print(f"\n[ai_director] ✅ Final reel: {out.name} ({mb:.1f} MB)")
    return out


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--no-ai", action="store_true")
    args = parser.parse_args()

    github_token = "" if args.no_ai else os.getenv("GITHUB_TOKEN", "")
    groq_key = "" if args.no_ai else os.getenv("GROQ_API_KEY", "")
    giphy_key = os.getenv("GIPHY_API_KEY", "")
    paths = [Path(p) for p in args.clips if Path(p).exists()]
    if not paths:
        print("No valid clips"); sys.exit(1)

    build_ai_reel(paths, github_token, groq_key, giphy_key)

"""
main.py — Master Orchestrator (v5 — Groq + OpenAI Edition)

Modes:
  --mode ai       GPT-4o + Groq + faster-whisper + Tenor + MoviePy  (default)
  --mode ffmpeg   Classic FFmpeg overlay (faster, no AI)
  --dry-run       Skip Google Drive upload

Full v5 flow (AI mode):
  1. sourcer.py     → Reddit .json bypass + yt-dlp + fallbacks
  2. ai_director.py → faster-whisper → GPT-4o → Groq → Giphy → MoviePy assembly
  3. uploader.py    → Upload .mp4 + .txt to Google Drive
"""

import os, sys, argparse
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR))

from sourcer import fetch_clips
from uploader import upload_reel_package


def run_ai_mode(raw_clips: list[Path]) -> Path | None:
    """Run GPT-4o + Groq + Whisper + Giphy + MoviePy pipeline."""
    from ai_director import build_ai_reel
    return build_ai_reel(
        raw_clips,
        github_token=os.getenv("GITHUB_TOKEN", ""),
        groq_key=os.getenv("GROQ_API_KEY", ""),
        giphy_key=os.getenv("GIPHY_API_KEY", ""),
    )


def run_ffmpeg_mode(raw_clips: list[Path], caption: str) -> Path | None:
    """Run the classic FFmpeg White Card overlay pipeline."""
    from card_generator import generate_card
    from processor import build_reel
    card_path, video_slot = generate_card(caption)
    return build_reel(raw_clips, card_path, video_slot)


def main(mode: str = "ai", dry_run: bool = False):
    print("=" * 60)
    print("  🐾 FUNNY ANIMAL REELS PIPELINE v5 (Groq + OpenAI)")
    print(f"  Mode: {mode.upper()}")
    print("=" * 60)

    pexels_key = os.getenv("PEXELS_API_KEY", "")
    pixabay_key = os.getenv("PIXABAY_API_KEY", "")

    # ── STEP 1: Source ───────────────────────────────────────────
    print("\n[STEP 1] Sourcing videos...")
    raw_clips = fetch_clips(
        pexels_key=pexels_key,
        pixabay_key=pixabay_key,
    )
    if not raw_clips:
        print("[main] ❌ No clips downloaded")
        sys.exit(1)
    print(f"[main] ✓ {len(raw_clips)} clips ready")

    # ── STEP 2: Process ─────────────────────────────────────────
    print(f"\n[STEP 2] Assembling reel ({mode} mode)...")
    if mode == "ai":
        output_video = run_ai_mode(raw_clips)
    else:
        # FFmpeg mode needs a simple caption
        from captioner import generate_caption, fallback_caption
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        caption = generate_caption(gemini_key) if gemini_key else fallback_caption()
        output_video = run_ffmpeg_mode(raw_clips, caption)

    if not output_video or not output_video.exists():
        print("[main] ❌ Video assembly failed")
        sys.exit(1)
    print(f"[main] ✓ Reel ready: {output_video.name}")

    # ── STEP 3: Upload ───────────────────────────────────────────
    caption_file = output_video.with_suffix(".txt")
    if dry_run:
        print("\n[STEP 3] DRY RUN — skipping Drive upload")
        print(f"  Video:   {output_video}")
        print(f"  Caption: {caption_file}")
    else:
        print("\n[STEP 3] Uploading to Google Drive...")
        try:
            result = upload_reel_package(output_video, caption_file)
            print(f"[main] ✓ Uploaded (folder_id={result['folder_id']})")
        except FileNotFoundError as e:
            print(f"[main] ⚠️  Drive skipped: {e}")

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Funny Animal Reels v5")
    parser.add_argument("--mode", choices=["ai", "ffmpeg"], default="ai")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(mode=args.mode, dry_run=args.dry_run)

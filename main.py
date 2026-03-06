"""
main.py — Master Orchestrator for Funny Animal Instagram Reels Pipeline (v2)

Modes:
  --mode ai       Use Gemini AI Director + MoviePy (default)
  --mode ffmpeg   Use the classic FFmpeg processor (faster, no AI)
  --dry-run       Skip Google Drive upload

Full v2 flow (AI mode):
  1. sourcer.py   → Download clips from Reddit/YT Shorts/Pexels/Pixabay
  2. card_generator.py → Generate White Meme Card PNG + determine video slot
  3. captioner.py → Generate caption + hashtags via Gemini
  4. ai_director.py → Analyse each clip with Gemini, assemble with MoviePy
  5. uploader.py  → Upload .mp4 + .txt to Google Drive "Ready for Make"
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from sourcer import fetch_clips
from captioner import generate_caption, fallback_caption, save_caption
from uploader import upload_reel_package


def run_ai_mode(raw_clips: list[Path], gemini_key: str, tenor_key: str) -> Path | None:
    """Run the Gemini AI Director + MoviePy assembly pipeline."""
    from card_generator import generate_card
    from ai_director import build_ai_reel
    return build_ai_reel(raw_clips, gemini_key, tenor_key)


def run_ffmpeg_mode(
    raw_clips: list[Path],
    caption: str,
) -> Path | None:
    """Run the classic FFmpeg White Card overlay pipeline."""
    from card_generator import generate_card
    from processor import build_reel

    card_path, video_slot = generate_card(caption)
    return build_reel(raw_clips, card_path, video_slot)


def main(mode: str = "ai", dry_run: bool = False):
    print("=" * 60)
    print("  🐾 FUNNY ANIMAL REELS PIPELINE v2")
    print(f"  Mode: {mode.upper()}")
    print("=" * 60)

    pexels_key = os.getenv("PEXELS_API_KEY", "")
    pixabay_key = os.getenv("PIXABAY_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    tenor_key = os.getenv("TENOR_API_KEY", "")

    # ── STEP 1: Source ───────────────────────────────────────────
    print("\n[STEP 1] Sourcing videos...")
    raw_clips = fetch_clips(pexels_key=pexels_key, pixabay_key=pixabay_key)
    if not raw_clips:
        print("[main] ❌ No clips downloaded — check your network or API keys")
        sys.exit(1)
    print(f"[main] ✓ {len(raw_clips)} clips ready")

    # ── STEP 2: Generate caption (used in both modes) ────────────
    print("\n[STEP 2] Generating caption...")
    caption_text = generate_caption(gemini_key) if gemini_key else fallback_caption()
    print(f"[main] Caption preview:\n{caption_text}\n" + "-" * 40)

    # ── STEP 3: Process video ────────────────────────────────────
    print(f"\n[STEP 3] Assembling reel ({mode} mode)...")
    if mode == "ai":
        output_video = run_ai_mode(raw_clips, gemini_key, tenor_key)
    else:
        output_video = run_ffmpeg_mode(raw_clips, caption_text)

    if not output_video or not output_video.exists():
        print("[main] ❌ Video assembly failed")
        sys.exit(1)

    print(f"[main] ✓ Reel ready: {output_video.name}")

    # ── STEP 4: Save caption alongside video ─────────────────────
    caption_file = save_caption(caption_text, output_video)

    # ── STEP 5: Upload to Google Drive ───────────────────────────
    if dry_run:
        print("\n[STEP 4] DRY RUN — skipping Drive upload")
        print(f"  Video:   {output_video}")
        print(f"  Caption: {caption_file}")
    else:
        print("\n[STEP 4] Uploading to Google Drive...")
        try:
            result = upload_reel_package(output_video, caption_file)
            print(f"[main] ✓ Uploaded to Drive (folder_id={result['folder_id']})")
        except FileNotFoundError as e:
            print(f"[main] ⚠️  Drive skipped: {e}")

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Funny Animal Reels Pipeline v2")
    parser.add_argument(
        "--mode", choices=["ai", "ffmpeg"], default="ai",
        help="ai = Gemini+MoviePy (default) | ffmpeg = classic FFmpeg overlay"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip Google Drive upload"
    )
    args = parser.parse_args()
    main(mode=args.mode, dry_run=args.dry_run)

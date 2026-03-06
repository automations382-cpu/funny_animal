"""
processor.py — FFmpeg Video Processing Module (v2 - White Card Mode)

New pipeline logic:
  1. Scale each raw clip to fit 1080px wide, preserving original aspect ratio
  2. Overlay the scaled video onto the center of the White Meme Card (PNG from card_generator.py)
  3. Interleave short meme inserts between clips
  4. Concatenate everything using the FFmpeg concat demuxer

No forced 9:16 crop on source video — preserve natural aspect ratio, pad on white card.
"""

import os
import sys
import shutil
import subprocess
import argparse
import random
from pathlib import Path
from datetime import datetime


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MEME_DIR = BASE_DIR / "meme_inserts"
PROCESSED_DIR = BASE_DIR / ".tmp" / "processed"
OUTPUT_DIR = BASE_DIR / ".tmp" / "output"

for d in [PROCESSED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Canvas
CARD_W = 1080
CARD_H = 1920
MAX_CLIP_DURATION = 10
MEME_EVERY_N_CLIPS = 2
MIN_TOTAL_DURATION = 15


def run_ffmpeg(args: list[str], desc: str = "") -> bool:
    """Run FFmpeg command. Returns True on success."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    print(f"[processor] {desc or 'ffmpeg...'}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[processor] FFmpeg error:\n{result.stderr[:600]}")
        return False
    return True


def get_video_info(path: Path) -> dict:
    """Return duration, width, height of a video via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        import json
        data = json.loads(result.stdout)
        streams = data.get("streams", [{}])
        fmt = data.get("format", {})
        w = streams[0].get("width", 1920) if streams else 1920
        h = streams[0].get("height", 1080) if streams else 1080
        dur = float(fmt.get("duration") or streams[0].get("duration") or 30)
        return {"width": w, "height": h, "duration": dur}
    except Exception:
        return {"width": 1920, "height": 1080, "duration": 30.0}


def overlay_clip_on_card(
    raw_clip: Path,
    card_png: Path,
    video_slot: tuple[int, int, int, int],
    dest: Path,
) -> bool:
    """
    Overlay a raw video clip onto the white meme card PNG.

    Steps:
      1. Scale the video to fit within slot_w × slot_h preserving aspect ratio
         using scale=w:h:force_original_aspect_ratio=decrease
      2. Pad to exactly slot_w × slot_h with white fill
         (so the video is always centered in the slot)
      3. Overlay the padded video onto the card image at (slot_x, slot_y)

    FFmpeg command structure:
      -loop 1 -i card.png            → static image input (looped)
      -i raw_clip.mp4                → video input
      -t {duration}                  → trim clip to MAX_CLIP_DURATION
      -filter_complex:
        [1:v]trim=duration={dur},
              scale={sw}:{sh}:force_original_aspect_ratio=decrease,
              pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:white,
              setsar=1,fps=30        → process the video track
        [0:v][vid]overlay={sx}:{sy}  → overlay onto card at (slot_x, slot_y)
    """
    sx, sy, sw, sh = video_slot
    info = get_video_info(raw_clip)
    dur = min(info["duration"], MAX_CLIP_DURATION)

    filter_chain = (
        f"[1:v]"
        f"trim=duration={dur:.2f},"
        f"scale={sw}:{sh}:force_original_aspect_ratio=decrease,"
        f"pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:white,"
        f"setsar=1,fps=30[vid];"
        f"[0:v][vid]overlay={sx}:{sy}[out]"
    )

    return run_ffmpeg([
        "-loop", "1", "-i", str(card_png),       # card background (static)
        "-i", str(raw_clip),                       # animal video
        "-t", str(dur),
        "-filter_complex", filter_chain,
        "-map", "[out]",
        "-map", "1:a?",                            # include audio if present
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dest)
    ], desc=f"Overlay → {dest.name}")


def process_meme_clip(src: Path, card_png: Path, video_slot: tuple, dest: Path) -> bool:
    """
    Scale a meme insert clip and overlay onto the same card template.
    Meme clips are capped at 3 seconds.
    """
    sx, sy, sw, sh = video_slot
    filter_chain = (
        f"[1:v]"
        f"trim=duration=3,"
        f"scale={sw}:{sh}:force_original_aspect_ratio=decrease,"
        f"pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:white,"
        f"setsar=1,fps=30[vid];"
        f"[0:v][vid]overlay={sx}:{sy}[out]"
    )
    return run_ffmpeg([
        "-loop", "1", "-i", str(card_png),
        "-i", str(src),
        "-t", "3",
        "-filter_complex", filter_chain,
        "-map", "[out]",
        "-map", "1:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        str(dest)
    ], desc=f"Meme overlay → {dest.name}")


def concatenate_clips(clip_paths: list[Path], output_path: Path) -> bool:
    """
    Concatenate all processed clips using the FFmpeg concat demuxer.

    All clips are already encoded to H.264 with same resolution,
    so we can stream-copy without re-encoding (-c copy).

    FFmpeg concat demuxer:
      -f concat -safe 0 -i list.txt  → reads clip list file
      -c copy                         → stream copy (no re-encode)
    """
    list_file = PROCESSED_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{str(p)}'\n")

    return run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path)
    ], desc=f"Concat {len(clip_paths)} clips → {output_path.name}")


def build_reel(
    raw_clips: list[Path],
    card_png: Path,
    video_slot: tuple[int, int, int, int],
) -> Path | None:
    """
    Main processing entrypoint.
    raw_clips: list of downloaded source .mp4s
    card_png: the white meme card PNG from card_generator
    video_slot: (x, y, w, h) position on the card where video should appear
    Returns: path to final output .mp4
    """
    if not raw_clips:
        print("[processor] ERROR: No raw clips provided")
        return None

    # ── Step 1: Overlay each animal clip onto the card ────────────
    processed_animal: list[Path] = []
    for i, clip in enumerate(raw_clips):
        dest = PROCESSED_DIR / f"animal_{i:02d}.mp4"
        if overlay_clip_on_card(clip, card_png, video_slot, dest):
            processed_animal.append(dest)
        else:
            print(f"[processor] Skipping {clip.name}")

    if not processed_animal:
        print("[processor] ERROR: All clips failed processing")
        return None

    # ── Step 2: Process meme clips on the same card ──────────────
    meme_clips = sorted(MEME_DIR.glob("*.mp4"))
    processed_memes: list[Path] = []
    for i, mc in enumerate(meme_clips):
        dest = PROCESSED_DIR / f"meme_{i:02d}.mp4"
        if process_meme_clip(mc, card_png, video_slot, dest):
            processed_memes.append(dest)

    # ── Step 3: Interleave animal + meme clips ────────────────────
    final_order: list[Path] = []
    meme_idx = 0
    for i, clip in enumerate(processed_animal):
        final_order.append(clip)
        if processed_memes and (i + 1) % MEME_EVERY_N_CLIPS == 0:
            final_order.append(processed_memes[meme_idx % len(processed_memes)])
            meme_idx += 1

    # ── Step 4: Pad to minimum duration if needed ─────────────────
    total_dur = sum(get_video_info(p)["duration"] for p in final_order)
    if total_dur < MIN_TOTAL_DURATION and processed_animal:
        print(f"[processor] Duration {total_dur:.1f}s < {MIN_TOTAL_DURATION}s — padding")
        extras = list(processed_animal)
        random.shuffle(extras)
        for clip in extras:
            if total_dur >= MIN_TOTAL_DURATION:
                break
            final_order.append(clip)
            total_dur += get_video_info(clip)["duration"]

    # ── Step 5: Concatenate ───────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"reel_{timestamp}.mp4"

    if not concatenate_clips(final_order, output_path):
        return None

    size_mb = output_path.stat().st_size / 1_048_576
    dur = get_video_info(output_path)["duration"]
    print(f"[processor] ✓ Final reel: {output_path.name} ({dur:.1f}s, {size_mb:.1f} MB)")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--card", type=str, default=None)
    args = parser.parse_args()

    card_path = Path(args.card) if args.card else PROCESSED_DIR / "meme_card.png"
    if not card_path.exists():
        print(f"[processor] Card PNG not found: {card_path} — run card_generator.py first")
        sys.exit(1)

    if args.clips:
        clips = [Path(p) for p in args.clips]
    else:
        clips = list((BASE_DIR / ".tmp" / "raw").glob("*.mp4"))[:3 if args.test else 8]

    # Default slot: center of card, 800px tall
    slot = (0, 560, CARD_W, 800)
    result = build_reel(clips, card_path, slot)
    if result:
        print(f"[processor] Output: {result}")
    else:
        sys.exit(1)

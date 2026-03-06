"""
sourcer.py — Video Sourcing Module (v7 — Pexels Primary + Robust YouTube)

Priority 1: Pexels API (Unblockable, high quality portrait videos)
Priority 2: YouTube Shorts (ytsearch5: with duration limits)

NO Reddit included due to absolute 403 blocks on GitHub Actions IPs.
"""

import os
import sys
import json
import random
import subprocess
import requests
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / ".tmp" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLIP_COUNT = 3
MIN_DURATION = 10
MAX_DURATION = 50

# Sourcing Queries
PEXELS_Q = ["funny animal fails", "cute pets"]
YT_QUERIES = ["funny animal shorts", "dog memes shorts", "cute cat fails shorts"]


def _dl(url: str, dest: Path) -> bool:
    """Helper to download a file with robust chunking."""
    if dest.exists() and dest.stat().st_size > 50_000:
        return True
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(262144):
                    f.write(chunk)
        if dest.stat().st_size > 50_000:
            return True
        else:
            dest.unlink()
            return False
    except Exception:
        if dest.exists(): dest.unlink()
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  1. Pexels API (Primary, Unblockable)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_pexels_clips(api_key: str, count: int) -> list[Path]:
    """Fetch high quality portrait videos from Pexels API."""
    if not api_key:
        print("[sourcer] No Pexels API Key provided.")
        return []
        
    print("[sourcer] Pexels: Starting integration search...")
    collected = []
    queries = list(PEXELS_Q)
    random.shuffle(queries)
    
    for q in queries:
        if len(collected) >= count:
            break
            
        print(f"[sourcer] Pexels: Searching '{q}' (portrait)...")
        headers = {"Authorization": api_key}
        params = {
            "query": q,
            "per_page": 15,
            "orientation": "portrait", # Reel-ready 9:16
            "size": "medium" # good balance of quality and size
        }
        
        try:
            r = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=20)
            r.raise_for_status()
            videos = r.json().get("videos", [])
            
            # Shuffle videos to get different results on subsequent runs
            random.shuffle(videos)
            
            for v in videos:
                if len(collected) >= count:
                    break
                    
                vid_id = str(v["id"])
                # Extract the best .mp4 file link
                mp4_url = None
                for f in sorted(v.get("video_files", []), key=lambda x: x.get("height", 9999), reverse=True): # highest quality first
                    if f.get("link", "").endswith(".mp4"):
                        mp4_url = f["link"]
                        break
                        
                if mp4_url:
                    dest = RAW_DIR / f"pexels_{vid_id}.mp4"
                    if _dl(mp4_url, dest):
                        collected.append(dest)
                        print(f"[sourcer]   ✓ Pexels downloaded: {dest.name}")
        except Exception as e:
            print(f"[sourcer] Pexels API error on '{q}': {e}")
            
    return collected


# ══════════════════════════════════════════════════════════════════════════════
#  2. YouTube Shorts (Failsafe)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yt_shorts(count: int) -> list[Path]:
    """Broad ytsearch5: fallback with yt-dlp"""
    if count <= 0: return []
    
    collected = []
    before = set(RAW_DIR.glob("*.mp4"))
    queries = list(YT_QUERIES)
    random.shuffle(queries)
    
    for query in queries:
        if len(collected) >= count:
            break
            
        print(f"[sourcer] YouTube: Search '{query}' ...")
        tpl = str(RAW_DIR / "yt_%(id)s.%(ext)s")
        # ytsearch{limit}: ensures it searches in the cloud
        cmd = [
            sys.executable, "-m", "yt_dlp", "--no-warnings", "--quiet",
            f"ytsearch5:{query}", "--output", tpl,
            "--match-filter", f"duration >= {MIN_DURATION} & duration <= {MAX_DURATION}",
            "--format", "mp4/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "--merge-output-format", "mp4", "--no-playlist", "--ignore-errors",
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            after = set(RAW_DIR.glob("yt_*.mp4"))
            new_files = list(after - before)
            for nf in new_files:
                if nf.stat().st_size > 50_000:
                    collected.append(nf)
                    before.add(nf) # add to 'before' so we don't count it twice
                    print(f"[sourcer]   ✓ YT downloaded: {nf.name}")
                    if len(collected) >= count:
                        break
        except Exception as e:
            print(f"[sourcer] YT error: {e}")
            
    return collected


# ══════════════════════════════════════════════════════════════════════════════
#  Master Coordinator
# ══════════════════════════════════════════════════════════════════════════════

def fetch_clips(
    user_agent: str = "", # Optional, kept since main.py might pass it in.
    pexels_key: str = "",
    pixabay_key: str = "",
    count: int = TARGET_CLIP_COUNT,
) -> list[Path]:
    """
    Ultimate Sourcing Flow (Reddit 403-proof version):
      1. Pexels (if key provided) -> primary because it's rarely blocked.
      2. YouTube Shorts -> robust fallback.
    """
    collected: list[Path] = []

    # 1. Pexels API
    try:
        if pexels_key:
            collected += fetch_pexels_clips(pexels_key, count)
    except Exception as e:
        print(f"[sourcer] Unhandled exception in fetch_pexels_clips: {e}")
        
    # 2. YouTube Shorts 
    try:
        if len(collected) < count:
            needed = count - len(collected)
            collected += fetch_yt_shorts(needed)
    except Exception as e:
        print(f"[sourcer] Unhandled exception in fetch_yt_shorts: {e}")

    collected = collected[:count]
    print(f"\n[sourcer] ✓ Sourcing Complete. Collected {len(collected)}/{count} clips.")
    return collected


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    
    clips = fetch_clips(
        pexels_key=os.getenv("PEXELS_API_KEY", ""),
        count=3,
    )
    print(json.dumps([str(p) for p in clips], indent=2))

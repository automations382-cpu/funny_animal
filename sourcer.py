"""
sourcer.py — Video Sourcing Module (v2)
Sources funny animal videos from:
  - Pexels API (free, API key required)
  - Pixabay API (free, API key required)
  - Reddit (r/funnyanimals, r/aww, etc.) via yt-dlp — NO API key needed
  - YouTube Shorts via yt-dlp — NO API key needed
Downloads raw .mp4 clips to .tmp/raw/
"""

import os
import sys
import json
import random
import argparse
import subprocess
import requests
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / ".tmp" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLIP_COUNT = 8
MAX_DURATION_SECS = 90

# Reddit subreddits to pull from (yt-dlp can scrape these natively)
REDDIT_SUBREDDITS = [
    "r/funnyanimals",
    "r/aww",
    "r/AnimalsBeingDerps",
    "r/catvideos",
    "r/dogvideos",
]

# YouTube Shorts search queries
YT_SHORTS_QUERIES = [
    "funny cat shorts",
    "funny dog fail shorts",
    "cute animal funny moment",
    "hilarious pet shorts",
]

PEXELS_QUERIES = ["funny cat", "funny dog", "funny animal", "cute animal fail"]
PIXABAY_QUERIES = ["funny animal", "funny pet", "cute dog funny", "funny cat video"]


# ── yt-dlp helpers ────────────────────────────────────────────────────────────

def _run_ytdlp(args: list[str], desc: str = "") -> dict | None:
    """Run yt-dlp as a subprocess and return parsed JSON, or None on failure."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings",
        "--quiet",
    ] + args
    if desc:
        print(f"[sourcer] {desc}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"[sourcer] yt-dlp stderr: {result.stderr[:300]}")
            return None
    except subprocess.TimeoutExpired:
        print("[sourcer] yt-dlp timed out")
        return None
    except Exception as e:
        print(f"[sourcer] yt-dlp error: {e}")
        return None


def fetch_reddit_clips(subreddit: str, limit: int = 5) -> list[Path]:
    """Download top short videos from a Reddit subreddit using yt-dlp."""
    url = f"https://www.reddit.com/{subreddit}/top/?t=day"
    output_template = str(RAW_DIR / "reddit_%(id)s.%(ext)s")

    result = _run_ytdlp([
        url,
        "--output", output_template,
        "--max-downloads", str(limit),
        "--match-filter", f"duration <= {MAX_DURATION_SECS}",
        "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--ignore-errors",
        "--extractor-args", "reddit:sort=top",
    ], desc=f"Scraping {subreddit} ...")

    # Return any newly created mp4s
    return _collect_new_mp4s(RAW_DIR, prefix="reddit_")


def fetch_youtube_shorts(query: str, limit: int = 4) -> list[Path]:
    """Search YouTube Shorts for a query and download short clips via yt-dlp."""
    search_url = f"ytsearch{limit}:{query} #shorts"
    output_template = str(RAW_DIR / "yt_%(id)s.%(ext)s")

    _run_ytdlp([
        search_url,
        "--output", output_template,
        "--match-filter", f"duration <= {MAX_DURATION_SECS}",
        "--format", "mp4/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--ignore-errors",
        "--extractor-args", "youtube:skip=translated_subs",
    ], desc=f"Searching YouTube Shorts: '{query}' ...")

    return _collect_new_mp4s(RAW_DIR, prefix="yt_")


def _collect_new_mp4s(directory: Path, prefix: str = "") -> list[Path]:
    """Return .mp4 files in directory matching an optional prefix."""
    files = sorted(directory.glob(f"{prefix}*.mp4"))
    return [f for f in files if f.stat().st_size > 50_000]


# ── Pexels / Pixabay helpers (fallback) ──────────────────────────────────────

def get_pexels_videos(api_key: str, query: str, per_page: int = 10) -> list[dict]:
    """Fetch video metadata from Pexels Videos API."""
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        clips = []
        for v in videos:
            files = sorted(v.get("video_files", []), key=lambda f: f.get("height", 9999))
            for f in files:
                if f.get("link", "").endswith(".mp4"):
                    clips.append({
                        "url": f["link"],
                        "duration": v.get("duration", 30),
                        "source": "pexels",
                        "id": str(v["id"]),
                    })
                    break
        return clips
    except Exception as e:
        print(f"[sourcer] Pexels error: {e}")
        return []


def get_pixabay_videos(api_key: str, query: str, per_page: int = 10) -> list[dict]:
    """Fetch video metadata from Pixabay Videos API."""
    url = "https://pixabay.com/api/videos/"
    params = {"key": api_key, "q": query, "per_page": per_page, "safesearch": "true"}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        clips = []
        for h in hits:
            videos = h.get("videos", {})
            for quality in ["medium", "small", "large"]:
                v = videos.get(quality, {})
                link = v.get("url", "")
                if link:
                    clips.append({
                        "url": link,
                        "duration": h.get("duration", 30),
                        "source": "pixabay",
                        "id": str(h["id"]),
                    })
                    break
        return clips
    except Exception as e:
        print(f"[sourcer] Pixabay error: {e}")
        return []


def download_clip(url: str, dest: Path) -> bool:
    """Stream-download a direct video URL to dest_path."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=262144):
                    f.write(chunk)
        return dest.stat().st_size > 50_000
    except Exception as e:
        print(f"[sourcer] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def fetch_api_clips(pexels_key: str, pixabay_key: str, count: int) -> list[Path]:
    """Download clips from Pexels / Pixabay APIs."""
    all_meta = []
    if pexels_key:
        query = random.choice(PEXELS_QUERIES)
        all_meta += get_pexels_videos(pexels_key, query, per_page=15)
    if pixabay_key:
        query = random.choice(PIXABAY_QUERIES)
        all_meta += get_pixabay_videos(pixabay_key, query, per_page=15)

    all_meta = [c for c in all_meta if c.get("duration", 30) <= MAX_DURATION_SECS]
    random.shuffle(all_meta)

    downloaded = []
    for clip in all_meta:
        if len(downloaded) >= count:
            break
        src = clip["source"]
        dest = RAW_DIR / f"{src}_{clip['id']}.mp4"
        if dest.exists() and dest.stat().st_size > 50_000:
            downloaded.append(dest)
            continue
        print(f"[sourcer] Downloading {src}:{clip['id']} ...")
        if download_clip(clip["url"], dest):
            downloaded.append(dest)
    return downloaded


# ── Main public function ──────────────────────────────────────────────────────

def fetch_clips(
    pexels_key: str = "",
    pixabay_key: str = "",
    count: int = TARGET_CLIP_COUNT,
) -> list[Path]:
    """
    Aggregate clips from all sources.
    Priority: Reddit > YouTube Shorts > Pexels/Pixabay fallback.
    Returns list of downloaded .mp4 paths.
    """
    collected: list[Path] = []

    # ── Source 1: Reddit via yt-dlp ──────────────────────────────
    subreddit = random.choice(REDDIT_SUBREDDITS)
    before = set(RAW_DIR.glob("*.mp4"))
    reddit_clips = fetch_reddit_clips(subreddit, limit=4)
    new_reddit = [p for p in reddit_clips if p not in before]
    collected += new_reddit
    print(f"[sourcer] Reddit → {len(new_reddit)} new clips from {subreddit}")

    # ── Source 2: YouTube Shorts via yt-dlp ─────────────────────
    if len(collected) < count:
        need = count - len(collected)
        yt_query = random.choice(YT_SHORTS_QUERIES)
        before_yt = set(RAW_DIR.glob("*.mp4"))
        yt_clips = fetch_youtube_shorts(yt_query, limit=min(need, 4))
        new_yt = [p for p in yt_clips if p not in before_yt and p not in collected]
        collected += new_yt
        print(f"[sourcer] YouTube Shorts → {len(new_yt)} new clips")

    # ── Source 3: Pexels / Pixabay (fallback) ───────────────────
    if len(collected) < count and (pexels_key or pixabay_key):
        need = count - len(collected)
        print(f"[sourcer] Fetching {need} more from Pexels/Pixabay ...")
        api_clips = fetch_api_clips(pexels_key, pixabay_key, need)
        for p in api_clips:
            if p not in collected:
                collected.append(p)

    collected = collected[:count]
    print(f"[sourcer] ✓ Total clips ready: {len(collected)}")
    return collected


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Fetch only 2 clips for quick test")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(BASE_DIR.parent / ".env")

    clips = fetch_clips(
        pexels_key=os.getenv("PEXELS_API_KEY", ""),
        pixabay_key=os.getenv("PIXABAY_API_KEY", ""),
        count=2 if args.test else TARGET_CLIP_COUNT,
    )
    print(json.dumps([str(p) for p in clips], indent=2))

"""
sourcer.py — Video Sourcing Module (v4 - Reddit .json Bypass)

NO praw. NO Reddit API keys.
Uses Reddit's public .json endpoint with a custom User-Agent header.

Flow:
  1. GET https://www.reddit.com/r/{sub}/top.json?t=day&limit=10
  2. Parse JSON → extract video URLs (v.redd.it links)
  3. Pass each URL to yt-dlp for high-quality .mp4 download

Fallbacks: YouTube Shorts → Pexels/Pixabay
"""

import os
import sys
import json
import random
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / ".tmp" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLIP_COUNT = 8
MAX_DURATION_SECS = 90

# Custom User-Agent — Reddit blocks generic requests/python agents.
# Format: platform:app_id:version (by /u/username)
REDDIT_USER_AGENT = "linux:madanimalx_scraper:v1.0 (by /u/automation)"

# Subreddits to scan
REDDIT_SUBREDDITS = [
    "funnyanimals",
    "AnimalsBeingDerps",
    "aww",
    "catvideos",
    "dogvideos",
]

YT_SHORTS_QUERIES = [
    "funny cat shorts",
    "funny dog fail shorts",
    "cute animal funny moment",
    "hilarious pet shorts",
]
PEXELS_QUERIES = ["funny cat", "funny dog", "funny animal", "cute animal fail"]
PIXABAY_QUERIES = ["funny animal", "funny pet", "cute dog funny", "funny cat video"]


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 1: Reddit .json bypass (NO API keys, NO praw)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_reddit_json(subreddit: str, limit: int = 10) -> list[dict]:
    """
    Fetch top posts from a subreddit using the public .json endpoint.

    GET https://www.reddit.com/r/{sub}/top.json?t=day&limit=10
    MUST use a specific User-Agent or Reddit responds with 429/403.

    Returns list of post dicts with keys: url, title, score, permalink.
    """
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"t": "day", "limit": limit, "raw_json": 1}
    headers = {"User-Agent": REDDIT_USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            print(f"[sourcer] Reddit rate-limited (429) for r/{subreddit} — skipping")
            return []
        resp.raise_for_status()
        data = resp.json()

        posts = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            # Skip NSFW, non-video, and self-posts
            if post.get("over_18"):
                continue
            if not post.get("is_video") and "v.redd.it" not in post.get("url", ""):
                continue

            posts.append({
                "url": f"https://www.reddit.com{post['permalink']}",
                "title": post.get("title", ""),
                "score": post.get("score", 0),
                "domain": post.get("domain", ""),
            })

        # Sort by score descending — most viral first
        posts.sort(key=lambda p: p["score"], reverse=True)
        return posts[:limit]

    except requests.exceptions.RequestException as e:
        print(f"[sourcer] Reddit .json error for r/{subreddit}: {e}")
        return []


def download_with_ytdlp(url: str, output_dir: Path, prefix: str = "reddit") -> Optional[Path]:
    """
    Download a single video URL using yt-dlp.
    Returns path to the downloaded .mp4, or None on failure.
    """
    output_template = str(output_dir / f"{prefix}_%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings", "--quiet",
        url,
        "--output", output_template,
        "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--max-filesize", "50M",
        "--match-filter", f"duration <= {MAX_DURATION_SECS}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            new_files = sorted(output_dir.glob(f"{prefix}_*.mp4"), key=lambda p: p.stat().st_mtime)
            if new_files:
                latest = new_files[-1]
                if latest.stat().st_size > 50_000:
                    return latest
        else:
            stderr = result.stderr[:200] if result.stderr else ""
            print(f"[sourcer] yt-dlp failed: {stderr}")
    except subprocess.TimeoutExpired:
        print(f"[sourcer] yt-dlp timed out for {url[:60]}")
    except Exception as e:
        print(f"[sourcer] yt-dlp error: {e}")
    return None


def fetch_reddit_clips(count: int = 5) -> list[Path]:
    """
    Scan Reddit subreddits via .json, get top viral video posts,
    download via yt-dlp.
    """
    all_posts: list[dict] = []
    subs = list(REDDIT_SUBREDDITS)
    random.shuffle(subs)

    for sub in subs:
        print(f"[sourcer] Fetching r/{sub}/top.json (day) ...")
        posts = fetch_reddit_json(sub, limit=10)
        for p in posts:
            print(f"[sourcer]   ↳ \"{p['title'][:55]}\" (↑{p['score']})")
        all_posts.extend(posts)

    # Deduplicate by URL and sort by score
    seen = set()
    unique = []
    for p in all_posts:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)
    unique.sort(key=lambda p: p["score"], reverse=True)

    print(f"[sourcer] Found {len(unique)} unique video posts across Reddit")

    # Download top N
    downloaded: list[Path] = []
    before_files = set(RAW_DIR.glob("*.mp4"))

    for post in unique[:count * 2]:  # try more than needed in case some fail
        if len(downloaded) >= count:
            break
        path = download_with_ytdlp(post["url"], RAW_DIR, prefix="reddit")
        if path and path not in before_files:
            downloaded.append(path)
            print(f"[sourcer]   ✓ Downloaded → {path.name}")

    print(f"[sourcer] Reddit (.json) → {len(downloaded)} clips downloaded")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 2: YouTube Shorts via yt-dlp
# ══════════════════════════════════════════════════════════════════════════════

def fetch_youtube_shorts(query: str, limit: int = 4) -> list[Path]:
    """Search YouTube Shorts and download clips via yt-dlp."""
    search_url = f"ytsearch{limit}:{query} #shorts"
    output_template = str(RAW_DIR / "yt_%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings", "--quiet",
        search_url,
        "--output", output_template,
        "--match-filter", f"duration <= {MAX_DURATION_SECS}",
        "--format", "mp4/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--no-playlist", "--ignore-errors",
    ]
    print(f"[sourcer] Searching YouTube Shorts: '{query}' ...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"[sourcer] YouTube Shorts error: {e}")

    return [f for f in sorted(RAW_DIR.glob("yt_*.mp4")) if f.stat().st_size > 50_000]


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 3: Pexels / Pixabay (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def get_pexels_videos(api_key: str, query: str, per_page: int = 10) -> list[dict]:
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        clips = []
        for v in resp.json().get("videos", []):
            files = sorted(v.get("video_files", []), key=lambda f: f.get("height", 9999))
            for f in files:
                if f.get("link", "").endswith(".mp4"):
                    clips.append({"url": f["link"], "duration": v.get("duration", 30),
                                  "source": "pexels", "id": str(v["id"])})
                    break
        return clips
    except Exception as e:
        print(f"[sourcer] Pexels error: {e}")
        return []


def get_pixabay_videos(api_key: str, query: str, per_page: int = 10) -> list[dict]:
    url = "https://pixabay.com/api/videos/"
    params = {"key": api_key, "q": query, "per_page": per_page, "safesearch": "true"}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        clips = []
        for h in resp.json().get("hits", []):
            videos = h.get("videos", {})
            for quality in ["medium", "small", "large"]:
                v = videos.get(quality, {})
                if v.get("url"):
                    clips.append({"url": v["url"], "duration": h.get("duration", 30),
                                  "source": "pixabay", "id": str(h["id"])})
                    break
        return clips
    except Exception as e:
        print(f"[sourcer] Pixabay error: {e}")
        return []


def download_clip(url: str, dest: Path) -> bool:
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
    all_meta = []
    if pexels_key:
        all_meta += get_pexels_videos(pexels_key, random.choice(PEXELS_QUERIES), 15)
    if pixabay_key:
        all_meta += get_pixabay_videos(pixabay_key, random.choice(PIXABAY_QUERIES), 15)
    all_meta = [c for c in all_meta if c.get("duration", 30) <= MAX_DURATION_SECS]
    random.shuffle(all_meta)
    downloaded = []
    for clip in all_meta:
        if len(downloaded) >= count:
            break
        dest = RAW_DIR / f"{clip['source']}_{clip['id']}.mp4"
        if dest.exists() and dest.stat().st_size > 50_000:
            downloaded.append(dest)
            continue
        print(f"[sourcer] Downloading {clip['source']}:{clip['id']} ...")
        if download_clip(clip["url"], dest):
            downloaded.append(dest)
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
#  Main public function
# ══════════════════════════════════════════════════════════════════════════════

def fetch_clips(
    pexels_key: str = "",
    pixabay_key: str = "",
    count: int = TARGET_CLIP_COUNT,
) -> list[Path]:
    """
    Aggregate clips from all sources.
    Priority: Reddit (.json bypass) → YouTube Shorts → Pexels/Pixabay.
    """
    collected: list[Path] = []

    # ── Source 1: Reddit .json (no API keys needed) ──────────────
    reddit_clips = fetch_reddit_clips(count=5)
    collected += reddit_clips

    # ── Source 2: YouTube Shorts ─────────────────────────────────
    if len(collected) < count:
        need = count - len(collected)
        yt_query = random.choice(YT_SHORTS_QUERIES)
        before_yt = set(RAW_DIR.glob("*.mp4"))
        yt_clips = fetch_youtube_shorts(yt_query, limit=min(need, 4))
        new_yt = [p for p in yt_clips if p not in before_yt and p not in collected]
        collected += new_yt
        print(f"[sourcer] YouTube Shorts → {len(new_yt)} new clips")

    # ── Source 3: Pexels / Pixabay ───────────────────────────────
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
    parser.add_argument("--test", action="store_true", help="Fetch only 2 clips")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(BASE_DIR.parent / ".env")

    clips = fetch_clips(
        pexels_key=os.getenv("PEXELS_API_KEY", ""),
        pixabay_key=os.getenv("PIXABAY_API_KEY", ""),
        count=2 if args.test else TARGET_CLIP_COUNT,
    )
    print(json.dumps([str(p) for p in clips], indent=2))

"""
sourcer.py — Video Sourcing Module (v3 - Praw + yt-dlp)

Sources funny animal videos from:
  1. Reddit via praw (authenticated API) — Top 5 most upvoted video posts
     from the last 24 hours across r/funnyanimals, r/AnimalsBeingDerps, etc.
  2. YouTube Shorts via yt-dlp — NO API key needed (fallback)
  3. Pexels / Pixabay — free API fallback

Priority: Reddit (praw) → YouTube Shorts → Pexels/Pixabay
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
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / ".tmp" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLIP_COUNT = 8
MAX_DURATION_SECS = 90

# Reddit subreddits to scan for viral animal content
REDDIT_SUBREDDITS = [
    "funnyanimals",
    "AnimalsBeingDerps",
    "aww",
    "catvideos",
    "dogvideos",
]

# YouTube Shorts search queries (fallback)
YT_SHORTS_QUERIES = [
    "funny cat shorts",
    "funny dog fail shorts",
    "cute animal funny moment",
    "hilarious pet shorts",
]

PEXELS_QUERIES = ["funny cat", "funny dog", "funny animal", "cute animal fail"]
PIXABAY_QUERIES = ["funny animal", "funny pet", "cute dog funny", "funny cat video"]


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 1: Reddit via praw (authenticated, precise, viral-proven)
# ══════════════════════════════════════════════════════════════════════════════

def _init_praw() -> Optional[object]:
    """
    Initialize praw Reddit instance from environment variables.
    Required env vars:
        REDDIT_CLIENT_ID
        REDDIT_CLIENT_SECRET
        REDDIT_USER_AGENT  (e.g. "madanimalx-bot/1.0 by u/your_username")
    Returns praw.Reddit instance, or None if credentials missing.
    """
    try:
        import praw
    except ImportError:
        print("[sourcer] praw not installed — run: pip install praw")
        return None

    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent = os.getenv("REDDIT_USER_AGENT", "madanimalx-bot/1.0")

    if not client_id or not client_secret:
        print("[sourcer] REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — skipping praw")
        return None

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        # Test the connection (read-only mode)
        reddit.read_only = True
        print(f"[sourcer] ✓ praw authenticated (read-only, user_agent={user_agent})")
        return reddit
    except Exception as e:
        print(f"[sourcer] praw init failed: {e}")
        return None


def fetch_reddit_viral(reddit_instance, subreddit_name: str, limit: int = 5) -> list[str]:
    """
    Fetch the top `limit` most upvoted video posts from the last 24 hours
    in a given subreddit using praw.

    Returns a list of direct Reddit post URLs that contain video.
    These URLs are then passed to yt-dlp for downloading.
    """
    urls = []
    try:
        subreddit = reddit_instance.subreddit(subreddit_name)
        # "top" with time_filter="day" = top posts in the last 24 hours
        for submission in subreddit.top(time_filter="day", limit=limit * 2):
            # Only keep posts that are actual videos
            is_video = (
                submission.is_video
                or "v.redd.it" in (submission.url or "")
                or submission.domain in ("v.redd.it", "youtube.com", "youtu.be")
            )
            if is_video and not submission.over_18:
                urls.append(f"https://www.reddit.com{submission.permalink}")
                print(f"[sourcer]   ↳ r/{subreddit_name}: \"{submission.title[:60]}\" "
                      f"(↑{submission.score})")
                if len(urls) >= limit:
                    break
    except Exception as e:
        print(f"[sourcer] Error scanning r/{subreddit_name}: {e}")
    return urls


def download_with_ytdlp(url: str, output_dir: Path, prefix: str = "reddit") -> Optional[Path]:
    """
    Download a single video URL using yt-dlp.
    Returns the path to the downloaded .mp4, or None on failure.
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
        f"--match-filter", f"duration <= {MAX_DURATION_SECS}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Find the newly created file
            new_files = sorted(output_dir.glob(f"{prefix}_*.mp4"), key=lambda p: p.stat().st_mtime)
            if new_files:
                latest = new_files[-1]
                if latest.stat().st_size > 50_000:
                    return latest
        else:
            stderr = result.stderr[:200] if result.stderr else ""
            print(f"[sourcer] yt-dlp failed for {url[:60]}: {stderr}")
    except subprocess.TimeoutExpired:
        print(f"[sourcer] yt-dlp timed out for {url[:60]}")
    except Exception as e:
        print(f"[sourcer] yt-dlp error: {e}")
    return None


def fetch_praw_clips(limit_per_sub: int = 5) -> list[Path]:
    """
    Main praw sourcing function.
    Scans multiple subreddits, fetches top viral video URLs,
    and downloads them via yt-dlp.
    Returns list of downloaded .mp4 paths.
    """
    reddit = _init_praw()
    if reddit is None:
        return []

    all_urls: list[str] = []
    # Shuffle subreddits for variety across runs
    subs = list(REDDIT_SUBREDDITS)
    random.shuffle(subs)

    for sub in subs:
        print(f"[sourcer] Scanning r/{sub} (top/day) ...")
        urls = fetch_reddit_viral(reddit, sub, limit=limit_per_sub)
        all_urls.extend(urls)

    # Deduplicate
    all_urls = list(dict.fromkeys(all_urls))
    print(f"[sourcer] Found {len(all_urls)} viral video URLs from Reddit")

    # Download each URL
    downloaded: list[Path] = []
    before_files = set(RAW_DIR.glob("*.mp4"))

    for url in all_urls:
        path = download_with_ytdlp(url, RAW_DIR, prefix="reddit")
        if path and path not in before_files:
            downloaded.append(path)
            print(f"[sourcer]   ✓ Downloaded → {path.name}")

    print(f"[sourcer] Reddit (praw) → {len(downloaded)} clips downloaded")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 2: YouTube Shorts via yt-dlp (no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_youtube_shorts(query: str, limit: int = 4) -> list[Path]:
    """Search YouTube Shorts for a query and download short clips via yt-dlp."""
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
        "--no-playlist",
        "--ignore-errors",
    ]
    print(f"[sourcer] Searching YouTube Shorts: '{query}' ...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"[sourcer] YouTube Shorts error: {e}")

    return _collect_new_mp4s(RAW_DIR, prefix="yt_")


def _collect_new_mp4s(directory: Path, prefix: str = "") -> list[Path]:
    """Return .mp4 files in directory matching an optional prefix."""
    files = sorted(directory.glob(f"{prefix}*.mp4"))
    return [f for f in files if f.stat().st_size > 50_000]


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 3: Pexels / Pixabay APIs (free, fallback)
# ══════════════════════════════════════════════════════════════════════════════

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
    Priority: Reddit (praw, viral-proven) → YouTube Shorts → Pexels/Pixabay.
    Returns list of downloaded .mp4 paths.
    """
    collected: list[Path] = []

    # ── Source 1: Reddit via praw (top viral posts) ──────────────
    praw_clips = fetch_praw_clips(limit_per_sub=5)
    collected += praw_clips

    # ── Source 2: YouTube Shorts via yt-dlp ──────────────────────
    if len(collected) < count:
        need = count - len(collected)
        yt_query = random.choice(YT_SHORTS_QUERIES)
        before_yt = set(RAW_DIR.glob("*.mp4"))
        yt_clips = fetch_youtube_shorts(yt_query, limit=min(need, 4))
        new_yt = [p for p in yt_clips if p not in before_yt and p not in collected]
        collected += new_yt
        print(f"[sourcer] YouTube Shorts → {len(new_yt)} new clips")

    # ── Source 3: Pexels / Pixabay (fallback) ────────────────────
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

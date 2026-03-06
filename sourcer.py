"""
sourcer.py — Video Sourcing Module (v5 — Reddit .json Bypass)

NO praw. NO Reddit API keys.
Loads REDDIT_USER_AGENT from .env so it's easily configurable.

Flow:
  1. GET /r/{sub}/top.json?t=day&limit=10  (custom User-Agent from .env)
  2. Parse JSON → extract v.redd.it video post URLs
  3. yt-dlp downloads .mp4
  Fallbacks: YouTube Shorts → Pexels → Pixabay
"""

import os, sys, json, random, subprocess, argparse, requests
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / ".tmp" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLIP_COUNT = 8
MAX_DURATION = 90

SUBREDDITS = ["funnyanimals", "AnimalsBeingDerps", "aww", "catvideos", "dogvideos"]
YT_QUERIES = ["funny animal fails 2026", "dog memes shorts", "cute cat fails shorts"]
PEXELS_Q = ["funny cat", "funny dog", "funny animal"]
PIXABAY_Q = ["funny animal", "funny pet", "cute dog funny"]


# ══════════════════════════════════════════════════════════════════════════════
#  Reddit .json Bypass
# ══════════════════════════════════════════════════════════════════════════════

def fetch_reddit_json(subreddit: str, user_agent: str, limit: int = 10) -> list[dict]:
    """Fetch top video posts from a subreddit via public .json endpoint."""
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"t": "day", "limit": limit, "raw_json": 1}
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            print(f"[sourcer] Rate-limited on r/{subreddit}")
            return []
        resp.raise_for_status()
        posts = []
        for child in resp.json().get("data", {}).get("children", []):
            d = child.get("data", {})
            if d.get("over_18"):
                continue
            if not d.get("is_video") and "v.redd.it" not in d.get("url", ""):
                continue
            posts.append({
                "url": f"https://www.reddit.com{d['permalink']}",
                "title": d.get("title", ""),
                "score": d.get("score", 0),
            })
        posts.sort(key=lambda p: p["score"], reverse=True)
        return posts[:limit]
    except Exception as e:
        print(f"[sourcer] Reddit error r/{subreddit}: {e}")
        return []


def download_ytdlp(url: str, output_dir: Path, prefix: str = "reddit") -> Optional[Path]:
    """Download a video URL via yt-dlp. Returns .mp4 path or None."""
    tpl = str(output_dir / f"{prefix}_%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-warnings", "--quiet",
        url, "--output", tpl,
        "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4", "--no-playlist",
        "--max-filesize", "50M",
        "--match-filter", f"duration <= {MAX_DURATION}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        hits = sorted(output_dir.glob(f"{prefix}_*.mp4"), key=lambda p: p.stat().st_mtime)
        if hits and hits[-1].stat().st_size > 50_000:
            return hits[-1]
    except Exception as e:
        print(f"[sourcer] yt-dlp error: {e}")
    return None


def fetch_reddit_clips(user_agent: str, count: int = 5) -> list[Path]:
    """Scan multiple subreddits, download top viral videos."""
    all_posts: list[dict] = []
    subs = list(SUBREDDITS)
    random.shuffle(subs)
    for sub in subs:
        print(f"[sourcer] Scanning r/{sub}/top.json ...")
        posts = fetch_reddit_json(sub, user_agent)
        for p in posts:
            print(f"[sourcer]   ↳ \"{p['title'][:55]}\" (↑{p['score']})")
        all_posts.extend(posts)

    # Deduplicate, sort by score
    seen, unique = set(), []
    for p in all_posts:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)
    unique.sort(key=lambda p: p["score"], reverse=True)
    print(f"[sourcer] {len(unique)} unique video posts found")

    downloaded: list[Path] = []
    for post in unique[:count * 2]:
        if len(downloaded) >= count:
            break
        path = download_ytdlp(post["url"], RAW_DIR)
        if path:
            downloaded.append(path)
            print(f"[sourcer]   ✓ {path.name}")
    print(f"[sourcer] Reddit → {len(downloaded)} clips")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
#  YouTube Shorts / Pexels / Pixabay (fallbacks)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yt_shorts(query: str, limit: int = 4) -> list[Path]:
    tpl = str(RAW_DIR / "yt_%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-warnings", "--quiet",
        f"ytsearch{limit}:{query} #shorts", "--output", tpl,
        "--match-filter", f"duration <= {MAX_DURATION}",
        "--format", "mp4/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4", "--no-playlist", "--ignore-errors",
    ]
    print(f"[sourcer] YouTube Shorts: '{query}' ...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"[sourcer] YT error: {e}")
    return [f for f in sorted(RAW_DIR.glob("yt_*.mp4")) if f.stat().st_size > 50_000]


def _dl(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(262144):
                    f.write(chunk)
        return dest.stat().st_size > 50_000
    except Exception:
        if dest.exists(): dest.unlink()
        return False


def fetch_api_clips(pexels_key: str, pixabay_key: str, count: int) -> list[Path]:
    meta = []
    if pexels_key:
        q = random.choice(PEXELS_Q)
        try:
            r = requests.get("https://api.pexels.com/videos/search",
                             headers={"Authorization": pexels_key},
                             params={"query": q, "per_page": 15}, timeout=20)
            for v in r.json().get("videos", []):
                for f in sorted(v.get("video_files", []), key=lambda x: x.get("height", 9999)):
                    if f.get("link", "").endswith(".mp4"):
                        meta.append({"url": f["link"], "id": str(v["id"]), "src": "pexels"})
                        break
        except Exception:
            pass
    if pixabay_key:
        q = random.choice(PIXABAY_Q)
        try:
            r = requests.get("https://pixabay.com/api/videos/",
                             params={"key": pixabay_key, "q": q, "per_page": 15}, timeout=20)
            for h in r.json().get("hits", []):
                for qual in ["medium", "small"]:
                    u = h.get("videos", {}).get(qual, {}).get("url")
                    if u:
                        meta.append({"url": u, "id": str(h["id"]), "src": "pixabay"})
                        break
        except Exception:
            pass
    random.shuffle(meta)
    out = []
    for c in meta[:count]:
        dest = RAW_DIR / f"{c['src']}_{c['id']}.mp4"
        if (dest.exists() and dest.stat().st_size > 50_000) or _dl(c["url"], dest):
            out.append(dest)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_clips(
    user_agent: str = "",
    pexels_key: str = "",
    pixabay_key: str = "",
    count: int = TARGET_CLIP_COUNT,
) -> list[Path]:
    """Aggregate clips. Priority: Reddit → YouTube Shorts → Pexels/Pixabay."""
    ua = user_agent or "linux:madanimalx_scraper:v1.0 (by /u/automation)"
    collected: list[Path] = []

    # Reddit .json
    collected += fetch_reddit_clips(ua, count=5)

    # YouTube Shorts
    if len(collected) < count:
        before = set(RAW_DIR.glob("*.mp4"))
        random.shuffle(YT_QUERIES)
        for query in YT_QUERIES:
            if len(collected) >= count:
                break
            yt = fetch_yt_shorts(query, min(count - len(collected), 4))
            added = 0
            for p in yt:
                if p not in before and p not in collected:
                    collected.append(p)
                    added += 1
            if added > 0:
                print(f"[sourcer] Added {added} clips from YT query: '{query}'")
                break  # If we got at least one batch from YT, we can stop or keep going, let's break to save time.

    # Pexels / Pixabay
    if len(collected) < count and (pexels_key or pixabay_key):
        collected += [p for p in fetch_api_clips(pexels_key, pixabay_key, count - len(collected))
                      if p not in collected]

    collected = collected[:count]
    print(f"[sourcer] ✓ Total: {len(collected)} clips")
    return collected


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    clips = fetch_clips(
        user_agent=os.getenv("REDDIT_USER_AGENT", ""),
        pexels_key=os.getenv("PEXELS_API_KEY", ""),
        pixabay_key=os.getenv("PIXABAY_API_KEY", ""),
        count=2,
    )
    print(json.dumps([str(p) for p in clips], indent=2))

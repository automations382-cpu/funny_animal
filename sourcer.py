"""
sourcer.py — Video Sourcing Module (v6 — Ultimate Sourcing)

Priority 1: Pexels API (Unblockable, high quality portrait videos)
Priority 2: Reddit .json (Stealth headers)
Priority 3: YouTube Shorts (ytsearch5: with duration limits)
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

TARGET_CLIP_COUNT = 5  # Usually need 3, getting 5 to be safe
MIN_DURATION = 5
MAX_DURATION = 50

# Sourcing Queries
PEXELS_Q = ["funny dog", "cute cat", "animal fails", "funny animal", "funny pet"]
YT_QUERIES = ["funny animal fails 2026", "dog memes shorts", "cute cat fails shorts", "funny pet shorts"]
SUBREDDITS = ["funnyanimals", "AnimalsBeingDerps", "aww", "catvideos", "dogvideos"]


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
#  2. Reddit JSON (Stealth Fallback)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_reddit_clips(user_agent: str, count: int) -> list[Path]:
    """Scan multiple subreddits, download top viral videos via .json."""
    if count <= 0: return []
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    all_posts = []
    subs = list(SUBREDDITS)
    random.shuffle(subs)
    
    for sub in subs:
        print(f"[sourcer] Reddit: Scanning r/{sub}/top.json ...")
        url = f"https://www.reddit.com/r/{sub}/top.json"
        params = {"t": "day", "limit": 10, "raw_json": 1}
        
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                print(f"[sourcer] Reddit: Rate-limited (429) on r/{sub}")
                continue
            elif resp.status_code == 403:
                print(f"[sourcer] Reddit: Blocked (403) on r/{sub}. Headers might be flagged.")
                continue
                
            resp.raise_for_status()
            
            for child in resp.json().get("data", {}).get("children", []):
                d = child.get("data", {})
                if d.get("over_18"):
                    continue
                if not d.get("is_video") and "v.redd.it" not in d.get("url", ""):
                    continue
                all_posts.append({
                    "url": f"https://www.reddit.com{d['permalink']}",
                    "title": d.get("title", ""),
                    "score": d.get("score", 0),
                })
        except Exception as e:
            print(f"[sourcer] Reddit error r/{sub}: {e}")

    # Deduplicate and sort by score
    seen, unique = set(), []
    for p in all_posts:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)
    unique.sort(key=lambda p: p["score"], reverse=True)
    
    print(f"[sourcer] Reddit: {len(unique)} unique video posts found.")

    downloaded = []
    for post in unique[:count * 2]:
        if len(downloaded) >= count:
            break
            
        dest = RAW_DIR / f"reddit_{post['url'].split('/')[-2]}.mp4"
        if dest.exists() and dest.stat().st_size > 50_000:
            downloaded.append(dest)
            continue
            
        cmd = [
            sys.executable, "-m", "yt_dlp", "--no-warnings", "--quiet",
            post["url"], "--output", str(dest),
            "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "--merge-output-format", "mp4", "--no-playlist",
            "--max-filesize", "50M",
            "--match-filter", f"duration >= {MIN_DURATION} & duration <= {MAX_DURATION}",
        ]
        
        try:
            print(f"[sourcer] Reddit: Downloading '...{post['url'][-20:]}'")
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if dest.exists() and dest.stat().st_size > 50_000:
                downloaded.append(dest)
                print(f"[sourcer]   ✓ Reddit downloaded: {dest.name}")
        except Exception as e:
            print(f"[sourcer] yt-dlp error on Reddit video: {e}")
            
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
#  3. YouTube Shorts (Failsafe)
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
            f"ytsearch5:{query} #shorts", "--output", tpl,
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
    user_agent: str = "",
    pexels_key: str = "",
    pixabay_key: str = "",
    count: int = TARGET_CLIP_COUNT,
) -> list[Path]:
    """
    Ultimate Sourcing Flow:
      1. Pexels (if key provided) -> primary because it's rarely blocked.
      2. Reddit (stealth headers) -> secondary viral content.
      3. YouTube Shorts -> robust fallback.
    """
    ua = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    collected: list[Path] = []

    # 1. Pexels API
    if pexels_key:
        collected += fetch_pexels_clips(pexels_key, count)
        
    # 2. Reddit .json
    if len(collected) < count:
        needed = count - len(collected)
        collected += fetch_reddit_clips(ua, needed)
        
    # 3. YouTube Shorts 
    if len(collected) < count:
        needed = count - len(collected)
        collected += fetch_yt_shorts(needed)

    collected = collected[:count]
    print(f"\n[sourcer] ✓ Sourcing Complete. Collected {len(collected)}/{count} clips.")
    return collected


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    
    clips = fetch_clips(
        user_agent=os.getenv("REDDIT_USER_AGENT", ""),
        pexels_key=os.getenv("PEXELS_API_KEY", ""),
        count=3,
    )
    print(json.dumps([str(p) for p in clips], indent=2))

# Funny Animal Instagram Reels — Automated Pipeline

Fully automated, zero-cost Instagram Reels pipeline. Downloads free funny animal clips, processes them into 9:16 videos with watermark and meme inserts, generates meme captions, and uploads to Google Drive for scheduling with Make.com.

Runs twice daily via **GitHub Actions**.

---

## 📁 Files

```
funny_animal/
├── main.py                 # Master orchestrator — run this
├── sourcer.py              # Downloads clips from Pexels + Pixabay  
├── processor.py            # FFmpeg: crop/scale/concat/watermark
├── captioner.py            # Gemini Flash: caption generation
├── uploader.py             # Google Drive upload
├── generate_watermark.py   # Run once to create assets/watermark.png
├── requirements.txt
├── assets/
│   └── watermark.png       # Auto-generated (run generate_watermark.py)
├── meme_inserts/           # Add your short meme .mp4 clips here (2-3 sec)
└── .tmp/                   # Auto-created intermediate + output files
    ├── raw/                # Downloaded source clips
    ├── processed/          # FFmpeg intermediate clips  
    └── output/             # Final reel .mp4 + caption .txt
```

---

## Setup

### 1. API Keys — add to `.env`

```env
PEXELS_API_KEY=your_key_here        # pexels.com/api — free
PIXABAY_API_KEY=your_key_here       # pixabay.com/api/docs — free
GEMINI_API_KEY=your_key_here        # ai.google.dev — free tier
```

### 2. Google Drive Credentials

1. Google Cloud Console → Enable **Google Drive API**
2. Create **OAuth 2.0 Desktop App** → download `credentials.json` to project root
3. Run pipeline once locally — browser opens for consent → `token.json` auto-created

### 3. Install & Generate Watermark

```bash
source venv/bin/activate
pip install -r funny_animal/requirements.txt
python funny_animal/generate_watermark.py   # one-time
```

### 4. Add Meme Insert Clips (optional)

Drop 2–6 short `.mp4` clips (2–3 sec) into `funny_animal/meme_inserts/` to interleave between animal clips.

---

## Running

```bash
python funny_animal/main.py            # Full run
python funny_animal/main.py --dry-run  # Skip Drive upload (testing)
```

---

## GitHub Actions Setup

Add these **Secrets** in GitHub → Settings → Secrets → Actions:

| Secret | Value |
|--------|-------|
| `PEXELS_API_KEY` | Your Pexels key |
| `PIXABAY_API_KEY` | Your Pixabay key |
| `GEMINI_API_KEY` | Your Gemini key |
| `GOOGLE_CREDENTIALS_JSON` | Full content of `credentials.json` |
| `GOOGLE_TOKEN_JSON` | Full content of `token.json` |

Runs at **09:30 IST** and **21:30 IST** daily.

---

## FFmpeg Pipeline

| Step | Operation | Filter |
|------|-----------|--------|
| 1 | Trim to 8 sec | `-t 8` |
| 2 | 9:16 crop | `crop=ih*9/16:ih` |
| 3 | Scale | `scale=1080:1920` |
| 4 | Concatenate | concat demuxer |
| 5 | Watermark | `overlay=W-w-40:H-h-80` |

See `processor.py` for full inline FFmpeg documentation.

"""
uploader.py — Google Drive Upload Module
Authenticates via OAuth2 and uploads .mp4 + .txt to "Ready for Make" folder.
"""

import os
import sys
import json
import argparse
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_FOLDER_NAME = "Ready for Make"

BASE_DIR = Path(__file__).parent
CREDS_PATH = BASE_DIR.parent / "credentials.json"   # project root
TOKEN_PATH = BASE_DIR.parent / "token.json"          # project root


def authenticate() -> "googleapiclient.discovery.Resource":
    """
    Authenticate with Google Drive.
    - First run: opens browser for OAuth2 consent, saves token.json
    - Subsequent runs: loads token.json, refreshes if expired
    - CI: reads token.json from env var GOOGLE_TOKEN_JSON (GitHub Secret)
    """
    creds = None

    # CI mode: read token from environment variable (GitHub Actions Secret)
    token_json_env = os.getenv("GOOGLE_TOKEN_JSON")
    if token_json_env:
        token_data = json.loads(token_json_env)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        print("[uploader] Loaded credentials from GOOGLE_TOKEN_JSON env var")

    # Local mode: load from token.json file
    elif TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        print("[uploader] Loaded credentials from token.json")

    # Refresh expired token
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
        print("[uploader] Token refreshed")

    # First-time OAuth2 flow
    elif not creds or not creds.valid:
        if not CREDS_PATH.exists():
            raise FileNotFoundError(
                f"credentials.json not found at {CREDS_PATH}\n"
                "Download OAuth2 Desktop credentials from Google Cloud Console."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
        print("[uploader] ✓ New token saved to token.json")

    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, folder_name: str) -> str:
    """
    Find the 'Ready for Make' folder in Drive root, or create it if missing.
    Returns the folder ID.
    """
    query = (
        f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' "
        "and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])

    if folders:
        folder_id = folders[0]["id"]
        print(f"[uploader] Found Drive folder: '{folder_name}' (id={folder_id})")
    else:
        metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = service.files().create(body=metadata, fields="id").execute()
        folder_id = folder["id"]
        print(f"[uploader] Created Drive folder: '{folder_name}' (id={folder_id})")

    return folder_id


def upload_file(service, file_path: Path, folder_id: str, mime_type: str) -> str:
    """Upload a single file to the specified Drive folder. Returns file ID."""
    metadata = {"name": file_path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)
    uploaded = service.files().create(
        body=metadata, media_body=media, fields="id, webViewLink"
    ).execute()
    file_id = uploaded.get("id")
    link = uploaded.get("webViewLink", "")
    size_mb = file_path.stat().st_size / 1_048_576
    print(f"[uploader] ✓ Uploaded {file_path.name} ({size_mb:.1f} MB) → {link}")
    return file_id


def upload_reel_package(video_path: Path, caption_path: Path) -> dict:
    """
    Full upload flow: authenticate → find/create folder → upload both files.
    Returns dict with video_id and caption_id.
    """
    service = authenticate()
    folder_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)

    video_id = upload_file(service, video_path, folder_id, "video/mp4")
    caption_id = upload_file(service, caption_path, folder_id, "text/plain")

    return {"video_id": video_id, "caption_id": caption_id, "folder_id": folder_id}


# ── CLI test mode ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--caption", required=True)
    args = parser.parse_args()

    result = upload_reel_package(Path(args.video), Path(args.caption))
    print(f"[uploader] Upload complete: {result}")

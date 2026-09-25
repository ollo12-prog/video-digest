#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

# Resolution mode (VD_RES env). The single source of truth that COUPLES the
# yt-dlp download cap to the describe-frame width so they can never drift: there
# is no point extracting a describe frame wider than the source we downloaded.
#   "1280"  → 720p source, describe frames 1280 wide (native; the free default)
#   "1080p" → 1080p source, describe frames 1920 wide (native; sharper on dense text)
# INVARIANT: the benchmark ground-truth frames must be
# re-frozen at describe_width() whenever the mode changes — describe input and
# bench must always be the same resolution (bench_describe.py warns if they drift).
RES_MODE = os.environ.get("VD_RES", "1280")
_RES_MODES = {
    "1280":  {"describe_width": 1280, "dl_height": 720},
    "1080p": {"describe_width": 1920, "dl_height": 1080},
}


# yt-dlp needs a JS runtime for YouTube and only auto-enables deno; fall back to node.
_JS_RUNTIME = (["--js-runtimes", "node"]
               if shutil.which("deno") is None and shutil.which("node") else [])


def _mode() -> dict:
    m = _RES_MODES.get(RES_MODE)
    if m is None:
        raise SystemExit(f"unknown VD_RES={RES_MODE!r}; use one of {list(_RES_MODES)}")
    return m


def describe_width() -> int:
    """Describe-frame extraction width for the active mode (scan.py imports this)."""
    return _mode()["describe_width"]


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    # Manual captions (.en. / .en-US. / .en-GB.) beat auto-generated (.en-orig.):
    # rank by the first marker tier that matches, then alphabetically.
    tiers = ((".en.", ".en-US.", ".en-GB."), (".en-orig.",))
    for tier in tiers:
        preferred = [c for c in candidates if any(m in c.name for m in tier)]
        if preferred:
            return preferred[0]
    return candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def _clear_outputs(out_dir: Path) -> None:
    """Remove video.* leftovers from a previous run: a failed yt-dlp run must not
    silently reuse a stale download and digest the wrong video."""
    for old in out_dir.glob("video*"):
        if old.is_file():
            old.unlink()


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(out_dir)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        *_JS_RUNTIME,
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(out_dir)
    output_template = str(out_dir / "video.%(ext)s")

    # yt-dlp needs ffmpeg to MERGE the separate video+audio streams. It looks on
    # PATH, not our FFMPEG env — so tell it explicitly, or it leaves them unmerged
    # (video-only mp4 → the whisper audio path then finds no audio track).
    ff = os.environ.get("FFMPEG")
    ff_loc = ["--ffmpeg-location", str(Path(ff).parent)] if ff and Path(ff).exists() else []

    h = _mode()["dl_height"]
    fmt = ("ba/bestaudio" if audio_only
           else f"bv*[height<={h}]+ba/b[height<={h}]/bv+ba/b")
    cmd = [
        "yt-dlp",
        *_JS_RUNTIME,
        "-N", "8",
        "-f", fmt,
        *ff_loc,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    video = _pick_video(out_dir)
    if video is None:
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode}). "
            "If YouTube errors (403, 'Sign in to confirm'), update yt-dlp first: "
            "pip install -U yt-dlp"
        )

    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))

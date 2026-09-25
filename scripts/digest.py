#!/usr/bin/env python3
"""video-digest orchestrator: URL or local file -> digest.md.

Two tracks off one download:
  AUDIO: native captions (preferred) else local whisper.cpp
  VIDEO: interval sample -> VLM classify -> dedup -> VLM describe (scan.py)
         (skipped with --transcript-only: no video download, no VLM needed)
Join at merge -> a single timestamped digest with inline described screenshots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import acquire
import merge
import scan
import vtt


def run(source: str, out_dir: Path, force_whisper: bool = False,
        transcript_only: bool = False, start: float | None = None,
        end: float | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[digest] acquiring…", file=sys.stderr)
    got = None
    if transcript_only and acquire.is_url(source) and not force_whisper:
        got = acquire.fetch_captions(source, out_dir)  # no media download
        if not got.get("subtitle_path"):
            print("[digest] no captions — fetching audio for whisper", file=sys.stderr)
            got = None
    if got is None:
        got = acquire.download(source, out_dir, audio_only=transcript_only)
    video = got["video_path"]
    sub = got.get("subtitle_path")
    if not video and not sub:
        raise SystemExit("nothing to process: no captions and no media")

    # AUDIO TRACK — captions first, whisper fallback.
    if sub and not force_whisper:
        print(f"[digest] transcript from captions: {Path(sub).name}", file=sys.stderr)
        transcript = vtt.parse_vtt(sub)
    else:
        print("[digest] no captions — local whisper.cpp ASR", file=sys.stderr)
        import asr_whisper
        transcript = asr_whisper.transcribe(video, out_dir)
    transcript = vtt.filter_range(transcript, start, end)
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")

    # VIDEO TRACK — the proven multi-pass.
    visuals = [] if transcript_only else scan.scan(video, out_dir, start, end)

    # MERGE.
    digest_md = merge.build(transcript, visuals, got.get("info", {}))
    out_path = out_dir / "digest.md"
    out_path.write_text(digest_md, encoding="utf-8")
    print(f"[digest] done -> {out_path} "
          f"({len(transcript)} segments, {len(visuals)} visuals)", file=sys.stderr)
    return out_path


def _seconds(ts: str) -> float:
    """'90', '1:30' or '1:02:30' -> seconds."""
    secs = 0.0
    for part in ts.split(":"):
        secs = secs * 60 + float(part)
    return secs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="URL or local video -> digest.md")
    ap.add_argument("source")
    ap.add_argument("out_dir", nargs="?", default="digest_out")
    ap.add_argument("--whisper", action="store_true", help="force local ASR even if captions exist")
    ap.add_argument("--transcript-only", action="store_true",
                    help="captions/ASR only: no video download, no vision model")
    ap.add_argument("--start", type=_seconds, help="only this range: start (SS, MM:SS or HH:MM:SS)")
    ap.add_argument("--end", type=_seconds, help="only this range: end")
    a = ap.parse_args()
    run(a.source, Path(a.out_dir), force_whisper=a.whisper,
        transcript_only=a.transcript_only, start=a.start, end=a.end)

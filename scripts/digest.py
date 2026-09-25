#!/usr/bin/env python3
"""video-digest orchestrator: URL or local file -> digest.md.

Two tracks off one download:
  AUDIO: native captions (preferred) else local whisper.cpp
  VIDEO: scene-detect -> VLM classify -> dedup -> VLM describe (scan.py)
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


def run(source: str, out_dir: Path, force_whisper: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[digest] acquiring…", file=sys.stderr)
    got = acquire.download(source, out_dir)
    video = got["video_path"]
    if not video:
        raise SystemExit("no video to process (caption-only fetch?)")

    # AUDIO TRACK — captions first, whisper fallback.
    sub = got.get("subtitle_path")
    if sub and not force_whisper:
        print(f"[digest] transcript from captions: {Path(sub).name}", file=sys.stderr)
        transcript = vtt.parse_vtt(sub)
    else:
        print("[digest] no captions — local whisper.cpp ASR", file=sys.stderr)
        import asr_whisper
        transcript = asr_whisper.transcribe(video, out_dir)
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")

    # VIDEO TRACK — the proven multi-pass.
    visuals = scan.scan(video, out_dir)

    # MERGE.
    digest_md = merge.build(transcript, visuals, got.get("info", {}))
    out_path = out_dir / "digest.md"
    out_path.write_text(digest_md, encoding="utf-8")
    print(f"[digest] done -> {out_path} "
          f"({len(transcript)} segments, {len(visuals)} visuals)", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--whisper"]
    if not args:
        print("usage: digest.py <url-or-file> [<out_dir>] [--whisper]", file=sys.stderr)
        raise SystemExit(2)
    out = Path(args[1]) if len(args) > 1 else Path("digest_out")
    run(args[0], out, force_whisper="--whisper" in sys.argv)

#!/usr/bin/env python3
"""Stage 6: interleave described visuals into the timestamped transcript.

Walk transcript segments in time order; wherever a captured visual's timestamp
falls, insert a [VISUAL @ MM:SS] block with the screenshot + OCR + description.
One digest.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _ts(t: float) -> str:
    mm, ss = divmod(int(t), 60)
    return f"{mm:02d}:{ss:02d}"


def _visual_block(v: dict) -> str:
    lines = [f"\n#### 🖼 Visual @ {v['ts']} — {v.get('type', 'graphic')}",
             f"![{v['frame']}](frames/{v['frame']})"]
    if v.get("ocr"):
        lines.append(f"\n> **On-screen text:** {v['ocr']}")
    if v.get("description"):
        lines.append(f"\n{v['description']}")
    if v.get("reappears_at"):
        lines.append(f"\n*(re-appears at {', '.join(v['reappears_at'])})*")
    return "\n".join(lines) + "\n"


def build(transcript: list[dict], visuals: list[dict], info: dict) -> str:
    vis = sorted(visuals, key=lambda v: v["t"])
    segs = sorted(transcript, key=lambda s: s["start"])

    out: list[str] = []
    title = info.get("title") or "Video digest"
    out.append(f"# {title}\n")
    meta = []
    if info.get("uploader"):
        meta.append(info["uploader"])
    if info.get("duration"):
        meta.append(f"{int(info['duration']) // 60}:{int(info['duration']) % 60:02d}")
    if info.get("url"):
        meta.append(info["url"])
    if meta:
        out.append(" · ".join(str(m) for m in meta) + "\n")
    out.append(f"*{len(vis)} visuals · {len(segs)} transcript segments*\n\n---\n")

    vi = 0
    for seg in segs:
        # Drop in any visual whose timestamp precedes this segment's start.
        while vi < len(vis) and vis[vi]["t"] <= seg["start"]:
            out.append(_visual_block(vis[vi]))
            vi += 1
        out.append(f"**[{_ts(seg['start'])}]** {seg['text']}")

    # Any trailing visuals after the last segment.
    while vi < len(vis):
        out.append(_visual_block(vis[vi]))
        vi += 1

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: merge.py <transcript.json> <visuals.json> [<info.json>] [<out.md>]",
              file=sys.stderr)
        raise SystemExit(2)
    transcript = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    visuals = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    info = {}
    if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
        info = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    md = build(transcript, visuals, info)
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("digest.md")
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out} ({len(md)} chars)", file=sys.stderr)

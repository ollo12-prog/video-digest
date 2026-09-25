#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs emit rolling-duplicate cues (each line appears 2-3 times as it
scrolls). We dedupe consecutive identical cues and merge their time ranges.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    segments: list[dict] = []
    i = 0
    while i < len(lines):
        match = TS_RE.match(lines[i])
        if not match:
            i += 1
            continue

        start = _to_seconds(*match.groups()[:4])
        end = _to_seconds(*match.groups()[4:])
        i += 1

        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            cleaned = TAG_RE.sub("", lines[i]).strip()
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        cue_text = " ".join(cue_lines).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        i += 1

    return _dedupe(segments)


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse YouTube's rolling-window duplication.

    Auto-captions scroll: each cue repeats the tail of the previous one and adds
    a few new words (e.g. "...on everyone's mind" then "and the big question on
    everyone's mind is does..."). We trim each cue's overlapping head at the WORD
    level so a segment keeps its own start time but carries only its new words —
    a simple exact/prefix check (the old approach) misses the mid-line overlap
    and leaves the transcript triplicated. Matching runs against a rolling tail
    of ALL emitted words, not just the previous segment: a 3-cue sliding window
    ("A B C" / "B C D" / "C D E") overlaps two segments back once trimming
    shrinks them. A fully-contained cue just extends the previous segment's end.
    Disjoint cues (manual captions) are untouched (k=0)."""
    out: list[dict] = []
    tail: list[str] = []  # last emitted words, cross-segment overlap window
    TAIL = 50
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        cur_words = text.split()
        # largest k where the emitted tail's last k words == this cue's first k
        k = 0
        for i in range(1, min(len(tail), len(cur_words)) + 1):
            if tail[-i:] == cur_words[:i]:
                k = i
        new_words = cur_words[k:]
        if not new_words:
            if out:
                out[-1]["end"] = seg["end"]  # fully contained → extend prev
            continue
        out.append({"start": seg["start"], "end": seg["end"],
                    "text": " ".join(new_words)})
        tail = (tail + new_words)[-TAIL:]
    return out


def filter_range(
    segments: list[dict],
    start_seconds: float | None,
    end_seconds: float | None,
) -> list[dict]:
    """Return segments whose time range overlaps [start, end]."""
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float("-inf")
    hi = end_seconds if end_seconds is not None else float("inf")
    return [seg for seg in segments if seg["end"] >= lo and seg["start"] <= hi]


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        start = int(seg["start"])
        stamp = f"[{start // 60:02d}:{start % 60:02d}]"
        lines.append(f"{stamp} {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))

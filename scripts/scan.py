#!/usr/bin/env python3
"""Video track: interval -> novelty-gate -> classify -> global-dedup -> describe.

v1 keyframe design (2026-07-09, Codex-reviewed — see reference/keyframe-design.md):
  1. INTERVAL sample every SAMPLE_EVERY s (one ffmpeg fps pass) — the coverage
     backbone. Catches gradual drawing that fires no scene-cut (the failure that
     triggered the redesign: 15 min of whiteboard drawing produced 0 cuts).
  2. CUMULATIVE NOVELTY GATE: keep a thumb when its dHash is > NOVELTY hamming from
     the LAST KEPT thumb. This is the empirically-validated probe — it emits
     gradual-drawing milestones without chaining the whole draw into one frame.
  3. CLASSIFY survivors -> {is_graphic, type}; drop captions/none.
  4. GLOBAL DEDUP: dHash each survivor against ALL kept (catches A->B->A re-shows).
  5. DESCRIBE + OCR each distinct survivor.

v2 (not yet wired): scene-detect union as boundary anchors, max-gap heartbeat,
persistence instead of blanket drop, a second cheap visual signal. detect_cuts()
below stays for that.

Output: <out>/frames/hit_###_tMMSS.jpg + <out>/visuals.json
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import acquire
import vlm

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
CLASSIFY_WIDTH = 384    # P3 keep/drop is a coarse scene judgment, not text reading:
                        # resolution-insensitive, so keep it small — this is the hot
                        # per-cut path (hundreds of calls). Do NOT raise to match describe.
# DESCRIBE_WIDTH comes from acquire.describe_width() (VD_RES mode) at the extract
# site — it's coupled to the download cap so we never ask for more than we fetched.
SAMPLE_EVERY = 3.0      # v1 interval: one thumb every N s (single ffmpeg fps pass)
NOVELTY = 30            # keep a thumb if dHash > this from the LAST KEPT thumb.
                        # Same 256-bit scale as PHASH_HAMMING; the probe used >30.
MAX_GAP = 20.0          # heartbeat (s): force-emit when content changes every sample
                        # so nothing ever persists (dashboards, typing) — else that
                        # stream would emit nothing. Pairs with the persistence check.
DHASH_SIZE = 16         # dHash grid -> DHASH_SIZE**2 bits (256)
PHASH_HAMMING = 30      # <= this on 256-bit dHash => near-identical re-show.
# Tuned on the bench video: the one true near-identical pair sits at d=19, the
# nearest *distinct* pair at d=40 — 30 splits the gap. This collapses only
# literal re-shows; a page shown again at a different scroll stays a separate
# visual (its differing content is worth keeping; P5's OCR flags the repeat).
CLASSIFY_WORKERS = 6    # concurrent P3 calls against llama-swap
DROP_TYPES = {"caption", "none"}  # not real graphics (refinement 2)


def detect_cuts(video: str, threshold: float = 0.05, min_gap: float = 0.4
                ) -> list[float]:
    """Recall-first scene detection, min-gap deduped. Unused by v1's interval
    core; kept for v2 boundary anchors (abrupt cuts the interval grid aliases past)."""
    cmd = [FFMPEG, "-i", video, "-vf",
           f"scale=160:-1,select='gt(scene,{threshold})',showinfo",
           "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg scene-detect failed (exit {proc.returncode}): "
                         f"{proc.stderr.strip()[-300:]}")
    times = sorted(float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr))
    deduped: list[float] = []
    for t in times:
        if not deduped or t - deduped[-1] > min_gap:
            deduped.append(t)
    return deduped


def interval_sample(video: str, out_dir: Path, width: int,
                    start: float | None = None, end: float | None = None
                    ) -> list[tuple[float, Path]]:
    """v1 step 1: one thumb every SAMPLE_EVERY s in a single ffmpeg fps pass
    (far cheaper than N seeks). Returns (approx_timestamp, path) in order.
    Timestamps are start + i*SAMPLE_EVERY — good to ~±SAMPLE_EVERY for labels/re-extract.
    Optional [start, end] (seconds) restricts sampling to that range of the video."""
    rng = (["-ss", f"{start:.3f}"] if start else []) + (["-to", f"{end:.3f}"] if end else [])
    subprocess.run(
        [FFMPEG, "-y", *rng, "-i", video, "-vf",
         f"fps=1/{SAMPLE_EVERY},scale={width}:-1", str(out_dir / "s_%05d.jpg")],
        capture_output=True,
    )
    thumbs = sorted(out_dir.glob("s_*.jpg"))
    t0 = start or 0.0
    return [(t0 + i * SAMPLE_EVERY, p) for i, p in enumerate(thumbs)]


def extract(video: str, t: float, out_path: Path, width: int,
            offset: float = 0.0) -> bool:
    """Grab one frame at t+offset; True if a non-empty file landed (a seek near
    EOF or a broken stream can fail quietly — callers skip it, never hash it)."""
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{t + offset:.3f}", "-i", video,
         "-frames:v", "1", "-vf", f"scale={width}:-1", str(out_path)],
        capture_output=True,
    )
    return out_path.exists() and out_path.stat().st_size > 0


def dhash(path: Path, sz: int = DHASH_SIZE) -> int:
    """Difference hash via ffmpeg (no Pillow dep): sz**2 bits encoding the sign
    of each pixel's left->right gradient. Discriminates layout, not just overall
    brightness — an average hash collapses distinct 'dark slide with text' frames."""
    raw = subprocess.run(
        [FFMPEG, "-y", "-i", str(path), "-vf",
         f"scale={sz + 1}:{sz},format=gray", "-f", "rawvideo", "-"],
        capture_output=True,
    ).stdout
    w = sz + 1
    if len(raw) < w * sz:
        return 0
    bits = 0
    k = 0
    for r in range(sz):
        row = raw[r * w:(r + 1) * w]
        for c in range(sz):
            if row[c] < row[c + 1]:
                bits |= (1 << k)
            k += 1
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _fmt_ts(t: float) -> str:
    mm, ss = divmod(int(t), 60)
    return f"{mm:02d}:{ss:02d}"


_PERSON_RE = re.compile(
    r"^(a |an )?((medium|close|closeup|close-up|wide|full|head)[ -]?(shot|up)?|"
    r"portrait|photo|headshot)?\s*(of )?(a |an )?"
    r"(man|woman|person|people|guy|presenter|speaker)\b", re.I)


_PERSON_OCR_MAX = 25    # a presenter shot carries at most social chrome
                        # ("@nate.b.jones", "read more on substack"); real content
                        # slides run far longer. Above this, keep even if a person
                        # is in frame (a screenshare that happens to show someone).


def _is_person_shot(rec: dict) -> bool:
    """Backstop for the classify model's stubborn habit of labeling a
    talking-head / webcam frame "screenshot" (survives even the tightened prompt).
    A real screenshot carries substantial on-screen text; a presenter shot carries
    none or just a handle/CTA — so drop a low-text frame the describe pass opens by
    naming a person.
    ponytail: heuristic with a ceiling — the 25-char OCR cutoff and the person regex
    are tuned to these videos; a text-sparse screenshot dominated by a person is
    dropped too (rare). Real fix = a classify model that doesn't call a photo of a
    person a screenshot."""
    return (len(rec["ocr"].strip()) < _PERSON_OCR_MAX
            and rec["type"] in ("screenshot", "none", "unknown", "")
            and bool(_PERSON_RE.match(rec["description"].strip())))


def scan(video: str, out_dir: Path, start: float | None = None,
         end: float | None = None) -> list[dict]:
    thumbs = out_dir / "thumbs"
    frames = out_dir / "frames"
    thumbs.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)

    # 1. Interval sample (single ffmpeg pass) — the coverage backbone.
    print(f"[scan] 1: interval sampling every {SAMPLE_EVERY}s…", file=sys.stderr)
    seq = interval_sample(video, thumbs, width=CLASSIFY_WIDTH, start=start, end=end)
    print(f"[scan] {len(seq)} interval thumbs", file=sys.stderr)

    # 2. Novelty gate + persistence + heartbeat. Keep a frame that is novel (dHash
    #    > NOVELTY from the last COMMITTED frame) only if the NEXT sample confirms it
    #    persisted — a 1-sample flash (app-switch flicker, transient dialog, blank
    #    canvas between states) is dropped. A max-gap heartbeat force-emits when
    #    content changes every sample so nothing settles, else that stream emits
    #    nothing. (v2. A brief slide shown for a single sample is dropped here —
    #    that's what the deferred scene-detect union anchors would recover.)
    hashes = [dhash(p) for _, p in seq]
    novel: list[tuple[float, Path]] = []
    committed: int | None = None
    last_emit_t = seq[0][0] if seq else 0.0
    for i, (t, p) in enumerate(seq):
        h = hashes[i]
        if committed is not None and hamming(h, committed) <= NOVELTY:
            continue  # within NOVELTY of the last kept frame — not new
        persisted = i + 1 < len(seq) and hamming(hashes[i + 1], h) <= NOVELTY
        if persisted or (t - last_emit_t >= MAX_GAP) or i == len(seq) - 1:
            novel.append((t, p))
            committed = h
            last_emit_t = t
    print(f"[scan] {len(novel)} kept (novelty+persistence gate)", file=sys.stderr)

    # 3. Classify the novel thumbs concurrently; keep real graphics.
    print("[scan] 3: classifying (concurrent)…", file=sys.stderr)
    def _classify(i_tp):
        i, (t, p) = i_tp
        try:
            return i, t, vlm.classify(p)
        except Exception as e:
            print(f"[scan]   t={t:.1f} classify failed: {e}", file=sys.stderr)
            return i, t, {"is_graphic": False, "type": "none"}

    clustered: list[tuple[float, str]] = []
    with cf.ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
        for i, t, res in sorted(ex.map(_classify, enumerate(novel))):
            if res["is_graphic"] and res["type"] not in DROP_TYPES:
                clustered.append((t, res["type"]))
    print(f"[scan] {len(clustered)} classified as real graphics", file=sys.stderr)

    # 4. Global dedup: extract full-res, dHash each against ALL kept (A->B->A re-shows).
    print("[scan] 4: extracting full-res + global dedup…", file=sys.stderr)
    kept: list[dict] = []
    hashes: list[int] = []
    for t, typ in clustered:
        tmp = frames / f"_tmp_{int(t)}.jpg"
        if not extract(video, t, tmp, width=acquire.describe_width()):
            print(f"[scan]   t={t:.1f} full-res extract failed — skipped",
                  file=sys.stderr)
            continue
        h = dhash(tmp)
        dup_of = next((k for k, kh in zip(kept, hashes)
                       if hamming(h, kh) <= PHASH_HAMMING), None)
        if dup_of is not None:
            dup_of["reappears_at"].append(_fmt_ts(t))
            tmp.unlink(missing_ok=True)
            continue
        idx = len(kept)
        final = frames / f"hit_{idx:03d}_t{int(t)//60:02d}{int(t)%60:02d}.jpg"
        tmp.replace(final)
        rec = {"t": round(t, 2), "ts": _fmt_ts(t), "type": typ,
               "frame": final.name, "reappears_at": [], "phash": h}
        kept.append(rec)
        hashes.append(h)
    print(f"[scan] {len(kept)} distinct visuals after dedup", file=sys.stderr)

    # 5. Describe + OCR each survivor (few calls, quality model). Drop any
    #    talking-head frame the classifier waved through (goal: no filler visuals).
    print("[scan] 5: describing…", file=sys.stderr)
    survivors: list[dict] = []
    for rec in kept:
        try:
            d = vlm.describe(frames / rec["frame"])
        except Exception as e:
            print(f"[scan]   {rec['frame']} describe failed: {e}", file=sys.stderr)
            d = {"type": rec["type"], "ocr": "", "description": ""}
        rec["ocr"], rec["description"] = d["ocr"], d["description"]
        if d["type"] not in ("unknown", ""):
            rec["type"] = d["type"]
        del rec["phash"]  # not needed downstream
        if _is_person_shot(rec):
            (frames / rec["frame"]).unlink(missing_ok=True)
            print(f"[scan]   {rec['ts']} DROPPED (talking-head, no text)",
                  file=sys.stderr)
            continue
        survivors.append(rec)
        print(f"[scan]   {rec['ts']} {rec['type']}: {rec['description'][:60]}",
              file=sys.stderr)
    kept = survivors
    print(f"[scan] {len(kept)} visuals after talking-head filter", file=sys.stderr)

    (out_dir / "visuals.json").write_text(
        json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    return kept


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: scan.py <video.mp4> [<out_dir>]", file=sys.stderr)
        raise SystemExit(2)
    scan(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else Path("digest_out"))

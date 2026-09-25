#!/usr/bin/env python3
"""Local ASR fallback via whisper.cpp server (T-B).

Only used when a video has no usable captions. whisper.cpp is the specialist
for word/segment-level timestamps — the omni VLMs give coarse timing. Output is
normalized to the same {start,end,text} shape as the VTT path so merge is
source-agnostic.

Endpoint: whisper.cpp server's native `/inference` (multipart; response_format
verbose_json → segments with start/end in seconds). It is NOT OpenAI-shaped, so
we talk to whisper-server directly. If the server
isn't already up we spawn it on demand (rare path; avoids a silent connection
refusal mid-pipeline) and leave it running for reuse.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

# whisper-server's own port (a sidecar, not the VLM endpoint). Override via env.
WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8123")
WHISPER_BIN = os.environ.get("WHISPER_BIN") or shutil.which("whisper-server") or "whisper-server"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "")  # e.g. .../ggml-large-v3-turbo.bin
# Optional: pin whisper to a separate GPU so it doesn't contend with the VLM for VRAM.
WHISPER_GPU = os.environ.get("WHISPER_GPU")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


_proc: subprocess.Popen | None = None  # set only if WE spawned the server


def _up() -> bool:
    try:
        requests.get(WHISPER_URL + "/", timeout=2)
        return True
    except requests.RequestException:
        return False


def ensure_server() -> None:
    """Start whisper-server if it isn't already listening; wait for model load."""
    global _proc
    if _up():
        return
    port = WHISPER_URL.rsplit(":", 1)[-1]
    if not Path(WHISPER_BIN).exists() or not WHISPER_MODEL:
        raise SystemExit(
            f"nothing is listening on {WHISPER_URL} and whisper-server can't be started: "
            f"set WHISPER_BIN (found: {WHISPER_BIN}) and WHISPER_MODEL. See docs/WHISPER.md.")
    print(f"[whisper] starting whisper-server on :{port}…", file=sys.stderr)
    env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": WHISPER_GPU} if WHISPER_GPU else {})}
    _proc = subprocess.Popen(
        [WHISPER_BIN, "-m", WHISPER_MODEL, "--host", "127.0.0.1",
         "--port", port, "-t", "8"],
        cwd=str(Path(WHISPER_BIN).parent),
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(90):  # model load can take a bit
        if _up():
            print("[whisper] server ready", file=sys.stderr)
            return
        time.sleep(1)
    raise SystemExit("whisper-server did not come up in time")


def stop_server() -> None:
    """Optional cleanup: stop a server WE spawned (left warm across runs by
    default). A pre-existing user-run server is left alone."""
    global _proc
    if _proc is not None:
        _proc.kill()
        _proc = None
        print("[whisper] stopped", file=sys.stderr)


def _segments(data: dict) -> list[dict]:
    """Normalize verbose_json -> [{start,end,text}] (start/end in seconds)."""
    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start") or 0.0), 2),
            "end": round(float(seg.get("end") or 0.0), 2),
            "text": text,
        })
    if not out and (data.get("text") or "").strip():
        out.append({"start": 0.0, "end": 0.0, "text": data["text"].strip()})
    return out


def extract_wav(video: str, out_path: Path) -> Path:
    """whisper.cpp wants 16 kHz mono PCM WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [FFMPEG, "-y", "-i", str(Path(video).resolve()),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(out_path.resolve())],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg audio extract failed: {r.stderr.strip()[:300]}")
    return out_path


def transcribe(video: str, work_dir: Path) -> list[dict]:
    ensure_server()
    wav = extract_wav(video, work_dir / "audio.wav")
    print(f"[whisper] transcribing {wav.stat().st_size // 1024} kB…", file=sys.stderr)
    with wav.open("rb") as f:
        resp = requests.post(
            WHISPER_URL + "/inference",
            files={"file": (wav.name, f, "audio/wav")},
            data={"response_format": "verbose_json", "temperature": "0"},
            timeout=1800,  # local ASR of a long video can take a while
        )
    resp.raise_for_status()
    segs = _segments(resp.json())
    if not segs:
        raise SystemExit("whisper returned no segments")
    print(f"[whisper] {len(segs)} segments", file=sys.stderr)
    return segs  # left running on GPU{WHISPER_GPU}; call stop_server() to reclaim it


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: asr_whisper.py <video-or-audio> [<work_dir>]", file=sys.stderr)
        raise SystemExit(2)
    wd = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("digest_out")
    print(json.dumps(transcribe(sys.argv[1], wd), indent=2, ensure_ascii=False))

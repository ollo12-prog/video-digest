---
name: video-digest
version: "1.0.0"
description: Turn a YouTube URL or local video into one digest.md — a timestamped transcript interleaved with described, deduped screenshots of every real on-screen visual (slide / screenshot / UI / diagram), talking-head filler filtered out; or just the timestamped transcript (--transcript-only, seconds, no GPU). Runs on any local OpenAI-compatible vision endpoint (LM Studio / Ollama / llama.cpp), no cloud APIs required.
argument-hint: "<video-url-or-path> [out-dir]"
allowed-tools: Bash, Read
user-invocable: true
---

# /video-digest

Given a YouTube URL (or a local video file), produce a single **digest.md**: the
full timestamped transcript with `🖼 Visual @ MM:SS` blocks inlined wherever a
real graphic appeared on screen — each screenshot OCR'd + described, near-
duplicates collapsed, talking-head / filler frames filtered out.

Everything runs against a **local, OpenAI-compatible vision endpoint** (LM Studio,
Ollama, llama.cpp, …). No cloud keys. See [README.md](README.md) for one-time setup.

## Resolve `SKILL_DIR`

Every command runs a bundled script under `SKILL_DIR/scripts/`. Set `SKILL_DIR`
to the absolute path of the directory containing THIS SKILL.md.

## Preflight (once)

```bash
python3 "$SKILL_DIR/scripts/setup.py"
```

Checks `ffmpeg`, `yt-dlp`, and that a vision model answers at `$VLM_ENDPOINT`;
prints the exact fix for anything missing. Configure via env (defaults in brackets):

| Env | Default | What |
|-----|---------|------|
| `VLM_ENDPOINT` | `http://localhost:1234/v1/chat/completions` | any OpenAI-compatible chat endpoint serving a vision model |
| `CLASSIFY_MODEL` | `minicpm-v-45-q4-32k-vision` | model id your endpoint serves for keep/drop |
| `DESCRIBE_MODEL` | `minicpm-v-45-q4-32k-vision` | model id for OCR + description |
| `VD_RES` | `1280` | `1280` (720p source) or `1080p` (1080p source, sharper OCR) |
| `WHISPER_*` | — | only for caption-less videos; see [docs/WHISPER.md](docs/WHISPER.md) |

**Do not swap the model.** `minicpm-v-45` is the validated model for both passes.
Bigger models your endpoint may list tested worse here: on dense terminal frames,
Qwen3.5-9B, Qwen3.6-35B-A3B and Gemma-4-26B put verbatim text in the wrong field, and
Nemotron hallucinated. Only minicpm-v-45 kept OCR and description separate. Keep the
default even when the endpoint serves larger models. Only change it if the user
explicitly asks.

## Run

```bash
python3 "$SKILL_DIR/scripts/digest.py" "<url-or-file>" "<out-dir>"
# force local ASR even when captions exist:
python3 "$SKILL_DIR/scripts/digest.py" "<file>" "<out-dir>" --whisper
# transcript only (no video download, no vision model; seconds, not minutes):
python3 "$SKILL_DIR/scripts/digest.py" "<url-or-file>" "<out-dir>" --transcript-only
```

Add `--start 12:30 --end 18:00` (SS, MM:SS or HH:MM:SS) to digest only part of a video.

Use `--transcript-only` when the user just wants the text/captions, or the video is
talking-head content with nothing on screen worth capturing.

Then `Read <out-dir>/digest.md`. Progress prints to stderr (the classify pass is
the bulk of the time).

## Output

```
<out-dir>/
  digest.md          # transcript + inline described-visual blocks  ← the deliverable
  frames/hit_###_tMMSS.jpg   # deduped full-res captures
  transcript.json    # [{start,end,text}]
  visuals.json       # [{t,ts,type,frame,ocr,description,reappears_at}]
```

## Pipeline (`scripts/`)

| Stage | Script | What |
|-------|--------|------|
| acquire | `acquire.py` | yt-dlp → video.mp4 + captions (.vtt) + info.json |
| audio | `vtt.py` / `asr_whisper.py` | captions (timestamps kept) → segments, else local whisper.cpp |
| video | `scan.py` | interval sample → novelty+persistence gate → VLM classify → global dedup → VLM describe → talking-head filter |
| merge | `merge.py` | interleave visuals into transcript → digest.md |
| orchestrator | `digest.py` | ties it together |
| — | `vlm.py` | shared VLM client (classify + describe) |

The keyframe selection is interval-based (not scene-cut based), so it captures
gradual on-screen change — e.g. a whiteboard drawn over several minutes — that
frame-to-frame scene detection misses. Both VLM passes use one model, so the
whole video track fits a 16 GB GPU (or runs CPU-side on ~24 GB RAM).

## Tuning (`scan.py`)

`SAMPLE_EVERY` interval · `NOVELTY` dedup/keep threshold · `MAX_GAP` heartbeat ·
`CLASSIFY_WORKERS` concurrency. If precision drifts on a new creator, tweak these
+ the `vlm.py` prompts — not the architecture.

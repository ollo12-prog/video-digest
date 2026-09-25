#!/usr/bin/env python3
"""Local VLM client for the video track: P3 classify + P5 describe.

Both hit an OpenAI-compatible chat endpoint (VLM_ENDPOINT). Two jobs, one model by default:
- classify  (many calls): short JSON -> keep/drop + type
- describe  (few calls):  rich JSON -> type/ocr/description

Gotcha: with a *thinking* model and a small max_tokens, the whole budget can go to
reasoning_content and content comes back empty. We send
chat_template_kwargs.enable_thinking=false to disable it.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

import requests

# Any OpenAI-compatible chat/completions endpoint that serves a vision model:
# LM Studio (default :1234), llama.cpp llama-server (:8080), Ollama (:11434,
# /v1), llama-swap, or a cloud gateway. Point VLM_ENDPOINT at yours.
ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")

# Both passes use minicpm-v-45: in a dense-frame benchmark (full-screen terminals,
# code) it was the only tested model that reliably split verbatim OCR from summary
# (Qwen and Gemma variants field-collapsed or hallucinated), it was the fastest
# (~1.7s/frame), and it fits 16 GB. As the classifier it matched a 35B model's
# keep/drop decisions exactly on two test videos. One model, no swaps.
# Override per-pass with CLASSIFY_MODEL / DESCRIBE_MODEL env.
CLASSIFY_MODEL = "minicpm-v-45-q4-32k-vision"
DESCRIBE_MODEL = "minicpm-v-45-q4-32k-vision"

TYPES = ("slide", "screenshot", "ui", "diagram", "caption", "none")

CLASSIFY_PROMPT = (
    "This is one frame from a YouTube video that is normally a person talking to "
    "camera in a room. Classify what THIS frame shows. Reply with ONLY a compact "
    'JSON object, no prose: {"is_graphic": <true|false>, "type": '
    '"slide|screenshot|ui|diagram|caption|none"}. '
    "is_graphic is true only if a slide, screenshot, app UI, or diagram REPLACES "
    "or covers most of the camera view. A \"screenshot\" means a capture of a "
    "COMPUTER SCREEN (an app, webpage, document, terminal, or editor). A photo or "
    "video of a real person, face, or physical room/scene is NOT a screenshot and "
    'NOT a graphic -> is_graphic false, type "none", even if it fills the frame and '
    "even if the person is not the usual presenter. Use type \"caption\" for a bold "
    "kinetic text word/phrase popped over the normal talking-head shot (that is NOT "
    'a graphic -> is_graphic false). Use "none" for a plain camera shot of the '
    "person (even mid-gesture, even with a small lower-third bar)."
)

DESCRIBE_PROMPT = (
    "This frame is an on-screen visual from a video (a slide, screenshot, app UI, "
    "or diagram). Describe it for a reader who cannot see it. Reply with ONLY a "
    "compact JSON object, no prose:\n"
    '{"type": "slide|screenshot|ui|diagram", '
    '"description": "<1-2 sentences: what this shows and its point>", '
    '"ocr": "<all on-screen text, verbatim, top-to-bottom; \\"\\" if none>"}\n'
    "Transcribe text exactly as shown, including numbers and labels. Keep the "
    "description concrete. Output ONE minified line — no pretty-printing."
)


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _request(model: str, prompt: str, img_b64: str, max_tokens: int,
             thinking: bool = False, timeout: int = 120) -> dict:
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    resp = requests.post(ENDPOINT, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _call(model: str, prompt: str, img_b64: str, max_tokens: int,
          thinking: bool = False, timeout: int = 120) -> str:
    data = _request(model, prompt, img_b64, max_tokens, thinking, timeout)
    return data["choices"][0]["message"]["content"] or ""


_DECODER = json.JSONDecoder(strict=False)  # strict=False: allow literal newlines
                                           # inside strings (multi-line OCR)


def _extract_json(text: str) -> dict | None:
    """Pull the FIRST balanced {...} object out of a model reply, tolerantly.

    raw_decode parses exactly one object from the first '{' and ignores anything
    after it — which handles, in one shot: ```json fences (we skip to the '{'),
    trailing prose, and models that emit two concatenated objects (nemotron).
    A greedy regex can't: it grabs first-brace-to-last-brace and chokes on the
    extra data. strict=False tolerates raw newlines some models put in OCR."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = _DECODER.raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def classify(path: Path, model: str | None = None) -> dict:
    """P3: {is_graphic: bool, type: str}. Robust to non-JSON replies."""
    import os
    model = model or os.environ.get("CLASSIFY_MODEL", CLASSIFY_MODEL)
    raw = _call(model, CLASSIFY_PROMPT, _b64(path), max_tokens=40)
    data = _extract_json(raw)
    if data is not None:
        t = str(data.get("type", "none")).lower().strip()
        if t not in TYPES:
            t = "none"
        g = data.get("is_graphic", False)
        if isinstance(g, str):  # models sometimes emit "true"/"false" as strings
            g = g.strip().lower() in ("true", "yes", "1")
        return {"is_graphic": bool(g), "type": t}
    # Fallback: keyword scan if the model ignored the JSON instruction.
    low = raw.lower()
    is_graphic = ("true" in low) or low.strip().startswith("y")
    return {"is_graphic": is_graphic, "type": "unknown" if is_graphic else "none"}


def _salvage_field(raw: str, field: str) -> str:
    """Pull one JSON string value out of a reply that didn't parse (truncated /
    degenerate). Ultra-dense screenshots can push a quantized model past clean
    OCR into unicode garbage that never closes the JSON — but the OCR *head* is
    real, so keep it and cut the garbage (a run of 6+ high-codepoint chars)."""
    m = re.search(r'"' + field + r'"\s*:\s*"((?:\\.|[^"\\])*)', raw)
    if not m:
        return ""
    s = m.group(1)
    try:
        s = json.loads('"' + s + '"')  # unescape \n, \uXXXX, etc.
    except json.JSONDecodeError:
        pass
    cut = re.search(r"[\u2001-\uffff]{6,}", s)  # 6+ high-codepoint chars = garbage run
    if cut:
        s = s[:cut.start()]
    return s.strip()


def describe(path: Path, model: str | None = None) -> dict:
    """P5: {type, ocr, description}. Model overridable for benching."""
    import os
    model = model or os.environ.get("DESCRIBE_MODEL", DESCRIBE_MODEL)
    # Description comes BEFORE ocr in the prompt: on very dense frames (2.5k+ chars)
    # the OCR can run on until max_tokens, and a description placed after it was
    # lost (6/32 empty on a coding video). More max_tokens doesn't help: the OCR
    # just runs longer. Description-first: 0/32 empty.
    raw = _call(model, DESCRIBE_PROMPT, _b64(path), max_tokens=1200)
    data = _extract_json(raw)
    if data is not None:
        return {
            "type": str(data.get("type", "unknown")).lower().strip(),
            "ocr": str(data.get("ocr", "")).strip(),
            "description": str(data.get("description", "")).strip(),
        }
    # Parse failed (truncated/degenerate) — salvage the real fields, don't dump
    # raw JSON into the digest.
    return {
        "type": (_salvage_field(raw, "type") or "unknown").lower(),
        "ocr": _salvage_field(raw, "ocr"),
        "description": _salvage_field(raw, "description"),
    }


def probe(job: str, path: Path, model: str, thinking: bool = False) -> None:
    """Manual debugging: send the EXACT skill prompt, dump the raw reply.

    Unlike classify/describe this hides nothing — you see finish_reason (the
    gemma non-termination shows as 'length'), reasoning_content (where thinking
    models dump output), token usage, and the unparsed content verbatim."""
    prompt, mt = ((CLASSIFY_PROMPT, 40) if job == "classify"
                  else (DESCRIBE_PROMPT, 1200))
    data = _request(model, prompt, _b64(path), mt, thinking=thinking, timeout=600)
    ch = data["choices"][0]
    msg = ch["message"]
    print(f"model:          {data.get('model', model)}")
    print(f"finish_reason:  {ch.get('finish_reason')}")
    print(f"usage:          {data.get('usage')}")
    rc = msg.get("reasoning_content")
    if rc:
        print(f"\n--- reasoning_content ({len(rc)} chars) ---\n{rc}")
    content = msg.get("content") or ""
    print(f"\n--- content ({len(content)} chars) ---\n{content}")


if __name__ == "__main__":
    # Skill-path check:   python vlm.py classify|describe <frame.jpg>
    # Raw model debug:    python vlm.py raw <frame.jpg> [model] [classify] [--thinking]
    #   (raw defaults to the describe prompt + DESCRIBE_MODEL; pass a model name
    #    to probe another, 'classify' to use the classify prompt instead)
    argv = [a for a in sys.argv[1:] if a != "--thinking"]
    job, frame = argv[0], Path(argv[1])
    if job == "raw":
        model = next((a for a in argv[2:] if a != "classify"), DESCRIBE_MODEL)
        kind = "classify" if "classify" in argv[2:] else "describe"
        probe(kind, frame, model, thinking="--thinking" in sys.argv)
    else:
        fn = {"classify": classify, "describe": describe}[job]
        print(json.dumps(fn(frame), indent=2, ensure_ascii=False))

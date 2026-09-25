#!/usr/bin/env python3
"""One-time preflight: verify ffmpeg, yt-dlp, python deps, and the vision endpoint.

Stdlib-only (urllib, not requests) so it runs before anything is pip-installed.
Prints the exact fix for whatever's missing; exit 0 iff the skill can run.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import urllib.request

ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
CLASSIFY_MODEL = os.environ.get("CLASSIFY_MODEL", "minicpm-v-45-q4-32k-vision")
DESCRIBE_MODEL = os.environ.get("DESCRIBE_MODEL", "minicpm-v-45-q4-32k-vision")

WIN, MAC = platform.system() == "Windows", platform.system() == "Darwin"
_INSTALL = {  # (macOS, Windows, Linux) one-liners
    "ffmpeg": ("brew install ffmpeg", "winget install ffmpeg",
               "sudo apt install ffmpeg  # or your distro's pkg mgr"),
    "yt-dlp": ("brew install yt-dlp", "winget install yt-dlp",
               "pipx install yt-dlp  # or: pip install yt-dlp"),
}


def _hint(tool: str) -> str:
    mac, win, lin = _INSTALL[tool]
    return mac if MAC else win if WIN else lin


def main() -> int:
    ok = True

    for tool in ("ffmpeg", "yt-dlp"):
        if shutil.which(tool):
            print(f"  ok    {tool}")
        else:
            ok = False
            print(f"  MISSING {tool} — install: {_hint(tool)}")

    try:
        import requests  # noqa: F401
        print("  ok    python: requests")
    except ImportError:
        ok = False
        print(f"  MISSING python 'requests' — install: {sys.executable} -m pip install requests")

    # Vision endpoint: list served models, check ours are there.
    models_url = ENDPOINT.rsplit("/chat/completions", 1)[0] + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as r:
            served = [m["id"] for m in json.load(r).get("data", [])]
        print(f"  ok    endpoint {ENDPOINT} ({len(served)} models)")
        for role, want in (("CLASSIFY_MODEL", CLASSIFY_MODEL), ("DESCRIBE_MODEL", DESCRIBE_MODEL)):
            if want in served:
                print(f"  ok    {role}={want}")
            else:
                ok = False
                print(f"  MISSING {role}={want} not served. Available: {served or '(none)'}")
                print(f"          → set {role} to one of the above (a vision model), e.g. export {role}=<id>")
    except Exception as e:
        ok = False
        print(f"  MISSING vision endpoint at {ENDPOINT} — {e}")
        print("          → start a local OpenAI-compatible server with a vision model")
        print("            (LM Studio: load minicpm-v-45, Developer → Start Server),")
        print("            or set VLM_ENDPOINT to your endpoint.")
        print("            If the server IS up but refused here, it's likely bound to your")
        print("            LAN IP (LM Studio 'Serve on Local Network' drops loopback) — try")
        print("            VLM_ENDPOINT=http://<your-LAN-IP>:1234/v1/chat/completions")

    print("\n" + ("All good — run: python3 scripts/digest.py <url-or-file> <out-dir>"
                  if ok else "Fix the MISSING items above, then re-run setup.py."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

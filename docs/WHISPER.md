# Local ASR fallback (whisper.cpp)

Only used when a video has no captions (most local files, or YouTube videos with
captions disabled). Videos with captions need none of this.

`asr_whisper.py` talks to [whisper.cpp](https://github.com/ggml-org/whisper.cpp)'s
`whisper-server` on its native `/inference` endpoint. If nothing is listening on
`WHISPER_URL`, it starts the server itself and leaves it running for later calls.

## Setup

1. Build or download `whisper-server` (whisper.cpp release, CUDA/Metal build recommended).
2. Download a model, e.g. `ggml-large-v3-turbo.bin` (~1.6 GB; much faster than
   large-v3 at similar accuracy).
3. Set the env vars below, or run the server yourself:

```bash
whisper-server -m /path/to/ggml-large-v3-turbo.bin --host 127.0.0.1 --port 8123
```

## Env

| Env | Default | What |
|-----|---------|------|
| `WHISPER_URL` | `http://127.0.0.1:8123` | server to use (or start) |
| `WHISPER_BIN` | `whisper-server` on PATH | binary to start if none is running |
| `WHISPER_MODEL` | — | model path; required only for auto-start |
| `WHISPER_GPU` | — | optional `CUDA_VISIBLE_DEVICES` for the server, to keep it off the VLM's GPU |

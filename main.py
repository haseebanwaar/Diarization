"""FastAPI backend for streaming ASR + diarization over websocket.

The websocket accepts 16 kHz mono PCM chunks, typically sent every 200 ms.
It emits compact JSON documents with both ASR and diarization fields so the
frontend can render partial and final text directly.
"""

from __future__ import annotations

import base64
import json
from typing import Any, cast

import nest_asyncio
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

nest_asyncio.apply()

app = FastAPI(title="Real-time ASR + Diarization")
cors_middleware = cast(Any, CORSMiddleware)
app.add_middleware(
    cors_middleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models at startup -------------------------------------------
from vad import VADWrapper
from pipeline import Pipeline

DEFAULT_PIPELINE_CONFIG = {
    "vad_threshold": 0.5,
    "silence_gap_sec": 0.6,
    "max_words": 24,
    "max_sentence_sec": 10.0,
    "partial_emit_sec": 0.2,
}


def build_pipeline() -> Pipeline:
    return Pipeline(vad=shared_vad, **DEFAULT_PIPELINE_CONFIG)


shared_vad = VADWrapper(threshold=DEFAULT_PIPELINE_CONFIG["vad_threshold"])


def _decode_audio_payload(payload) -> bytes:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)

    if isinstance(payload, str):
        return base64.b64decode(payload)

    if isinstance(payload, dict):
        if "audio" in payload:
            return _decode_audio_payload(payload["audio"])
        if "data" in payload:
            return _decode_audio_payload(payload["data"])

    raise ValueError("Unsupported audio payload format")


# --- HTTP endpoint (single-request, like the original) -----------------

@app.post("/diarize_live")
async def diarize_live(request: Request):
    """Debug-friendly HTTP bridge for one-shot audio payloads."""
    content_type = request.headers.get("content-type", "")
    pipe = build_pipeline()

    if "application/json" in content_type:
        payload = await request.json()
        raw = _decode_audio_payload(payload)
    else:
        raw = await request.body()

    async def generator():
        for event in pipe.feed(raw):
            yield json.dumps(event, ensure_ascii=False) + "\n"
        for event in pipe.finish():
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")


# --- WebSocket endpoint (continuous streaming) ------------------------

@app.websocket("/ws_stream")
async def ws_stream(websocket: WebSocket):
    """
    Continuous streaming via WebSocket.
    - Client sends binary PCM frames or JSON/base64 text frames.
    - Server sends compact JSON text frames.
    """
    await websocket.accept()
    local_pipe = build_pipeline()

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                break

            raw = None
            if message.get("bytes") is not None:
                raw = message["bytes"]
            elif message.get("text"):
                text = message["text"].strip()
                if text.lower() == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = text
                raw = _decode_audio_payload(payload)

            if raw is None:
                continue

            for event in local_pipe.feed(raw):
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
    except WebSocketDisconnect:
        # Flush remaining buffer on disconnect.
        for event in local_pipe.finish():
            try:
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
            except Exception:
                pass


# --- entrypoint -------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

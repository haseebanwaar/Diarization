# Diarization Streaming Backend

This backend exposes a websocket that accepts streaming audio chunks and returns
JSON events that already contain ASR and diarization fields.

## Websocket

### `WS /ws_stream`

Send:
- binary `int16` PCM audio frames, 16 kHz, mono
- or a text frame containing JSON with `audio` / `data` as base64 PCM

Recommended chunk size:
- ~200 ms per frame

Receive:
- JSON text frames with `type: "partial"` while speech is still open
- JSON text frames with `type: "final"` when the backend closes a sentence

### Event shape

```json
{
  "id": 1,
  "segment_id": 3,
  "type": "partial",
  "final": false,
  "reason": "streaming",
  "text": "hello world",
  "speaker": "unknown",
  "sentence_index": 0,
  "segment": {"start": 1.2, "end": 2.1, "duration": 0.9},
  "asr": {
    "text": "hello world",
    "full_text": "hello world",
    "word_count": 2
  },
  "diarization": {
    "status": "pending",
    "speaker": "unknown",
    "segments": []
  },
  "debug": {
    "stream_samples": 3200,
    "speaking": true,
    "chunk_ms": 200.0
  }
}
```

## Sentence control

The pipeline ends a sentence when any of these happen:
- VAD detects enough silence after speech
- the transcript exceeds the configured max word limit
- the segment exceeds the configured max duration
- the stream ends

This keeps sentences short and easier for a frontend to render directly.

## HTTP debug endpoint

### `POST /diarize_live`

Accepts the same audio payload as a one-shot request and streams newline-delimited
JSON (`application/x-ndjson`) back to the client.


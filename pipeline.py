"""Streaming ASR + diarization pipeline for websocket audio chunks.

The pipeline is intentionally stateful and debuggable:

* audio is received in arbitrary chunk sizes, typically 200 ms over websocket
* VAD runs on 512-sample frames to decide when speech starts/stops
* partial JSON events are emitted while a sentence is still open
* final JSON events are emitted when VAD falls silent or the segment grows too
  long (max words / max seconds)

The caller only needs to feed bytes or float32 PCM and forward the yielded
event dictionaries to the client.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from buffer import AudioBuffer
# from nemo_dia import nemo_dia
from parakeet import nemo_transcribe
from vad import VADWrapper

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512


def _pcm_to_float32(raw_pcm: bytes | np.ndarray) -> np.ndarray:
    if isinstance(raw_pcm, bytes):
        if not raw_pcm:
            return np.empty((0,), dtype=np.float32)
        pcm = np.frombuffer(raw_pcm, dtype=np.int16)
        return (pcm.astype(np.float32, copy=False) / np.float32(32768.0)).astype(np.float32, copy=False)

    audio = np.asarray(raw_pcm, dtype=np.float32)
    if audio.ndim != 1:
        return audio.reshape(-1).astype(np.float32)
    return audio


def _word_count(text: str) -> int:
    return len([word for word in text.strip().split() if word])


class Pipeline:
    """Stateful real-time pipeline with partial/final JSON events."""

    def __init__(
        self,
        vad: VADWrapper | None = None,
        asr_fn=None,
        # dia_fn=None,
        vad_threshold: float = 0.5,
        silence_gap_sec: float = 0.6,
        max_words: int = 24,
        max_sentence_sec: float = 10.0,
        partial_emit_sec: float = 0.8,
        min_segment_sec: float = 0.6,
        # min_partial_samples: int | None = None,
    ):
        self.vad = vad or VADWrapper(threshold=vad_threshold)
        self.asr = asr_fn or nemo_transcribe
        # self.dia = dia_fn or nemo_dia

        self.vad_threshold = vad_threshold
        self.silence_gap_samples = int(silence_gap_sec * SAMPLE_RATE)
        self.max_words = max_words
        self.max_sentence_samples = int(max_sentence_sec * SAMPLE_RATE)
        self.partial_emit_samples = max(1, int(partial_emit_sec * SAMPLE_RATE))
        self.min_segment_samples = int(min_segment_sec * SAMPLE_RATE)

        self._pending_audio = np.empty((0,), dtype=np.float32)
        self._stream_samples = 0
        self._speaking = False
        self._silence_samples = 0
        self._segment_start_sample = 0
        self._last_voice_sample = 0
        self._current_chunks: list[np.ndarray] = []
        self._segment_index = 0
        self._event_index = 0
        self._last_partial_text = ""
        self._last_partial_emit_sample = 0
        self._last_final_speaker = "unknown"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def feed(self, raw_pcm: bytes | np.ndarray) -> Iterator[dict]:
        """Consume one websocket chunk and yield zero or more events."""
        audio = _pcm_to_float32(raw_pcm)
        if audio.size == 0:
            return

        if self._pending_audio.size:
            audio = np.concatenate([self._pending_audio, audio])
            self._pending_audio = np.empty((0,), dtype=np.float32)

        full_frames = len(audio) // VAD_FRAME_SAMPLES
        for frame_idx in range(full_frames):
            start = frame_idx * VAD_FRAME_SAMPLES
            frame = audio[start : start + VAD_FRAME_SAMPLES]
            yield from self._process_frame(frame)

        remainder = len(audio) % VAD_FRAME_SAMPLES
        if remainder:
            self._pending_audio = audio[-remainder:].copy()

        if self._speaking:
            yield from self._maybe_emit_partial(force=False)

    def finish(self) -> Iterator[dict]:
        """Flush any open segment at stream end."""
        if self._pending_audio.size and self._speaking:
            yield from self._append_audio(self._pending_audio)
        self._pending_audio = np.empty((0,), dtype=np.float32)

        if self._speaking:
            yield from self._finalize_segment(reason="stream_end")

    # ------------------------------------------------------------------
    # frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> Iterator[dict]:
        is_speech = self.vad.is_speech(frame)
        frame_start_sample = self._stream_samples

        if is_speech:
            if not self._speaking:
                self._start_segment(frame_start_sample)

            yield from self._append_audio(frame)
            self._last_voice_sample = frame_start_sample + len(frame)
            self._silence_samples = 0

            if len(self._current_audio()) >= self.max_sentence_samples:
                yield from self._finalize_segment(reason="max_duration")

        elif self._speaking:
            self._silence_samples += len(frame)
            if self._silence_samples >= self.silence_gap_samples:
                yield from self._finalize_segment(reason="vad_silence")

        self._stream_samples += len(frame)

    def _start_segment(self, start_sample: int) -> None:
        self._segment_index += 1
        self._speaking = True
        self._silence_samples = 0
        self._segment_start_sample = start_sample
        self._last_voice_sample = start_sample
        self._current_chunks = []
        self._last_partial_text = ""
        self._last_partial_emit_sample = start_sample

    def _append_audio(self, frame: np.ndarray) -> Iterator[dict]:
        self._current_chunks.append(frame.astype(np.float32, copy=True))
        if self._speaking:
            yield from self._maybe_emit_partial(force=False)

    def _current_audio(self) -> np.ndarray:
        if not self._current_chunks:
            return np.empty((0,), dtype=np.float32)
        if len(self._current_chunks) == 1:
            return self._current_chunks[0]
        return np.concatenate(self._current_chunks)

    def _segment_duration_samples(self) -> int:
        return len(self._current_audio())

    def _segment_duration_sec(self) -> float:
        return self._segment_duration_samples() / SAMPLE_RATE

    # ------------------------------------------------------------------
    # partial / final emission
    # ------------------------------------------------------------------

    def _maybe_emit_partial(self, force: bool) -> Iterator[dict]:
        if not self._speaking:
            return

        segment = self._current_audio()
        if len(segment) < self.min_segment_samples and not force:
            return

        if not force and (self._stream_samples - self._last_partial_emit_sample) < self.partial_emit_samples:
            return

        text = (self.asr(segment) or "").strip()
        if not text:
            return

        words = _word_count(text)
        if words >= self.max_words or len(segment) >= self.max_sentence_samples:
            yield from self._finalize_segment(
                reason="max_words" if words >= self.max_words else "max_duration",
            )
            return

        if not force and text == self._last_partial_text:
            return

        self._last_partial_text = text
        self._last_partial_emit_sample = self._stream_samples
        speaker = self._last_final_speaker
        yield self._build_event(
            event_type="partial",
            text=text,
            full_text=text,
            speaker=speaker,
            # diarization={
            #     "status": "pending",
            #     "speaker": speaker,
            #     "segments": [],
            # },
            start_sample=self._segment_start_sample,
            end_sample=self._last_voice_sample,
            reason="streaming",
            sentence_index=0,
            final=False,
        )

    def _finalize_segment(self, reason: str) -> Iterator[dict]:
        if not self._speaking:
            return

        segment = self._current_audio()
        start_sample = self._segment_start_sample
        end_sample = max(self._last_voice_sample, start_sample + len(segment))

        self._speaking = False
        self._silence_samples = 0
        self._last_partial_text = ""
        self._last_partial_emit_sample = self._stream_samples
        self._current_chunks = []

        if len(segment) < self.min_segment_samples:
            return

        full_text = (self.asr(segment) or "").strip()
        if not full_text:
            return

        # raw_segments = self.dia(segment) or []
        # timeline = self._parse_diarization(raw_segments, segment)
        # primary_speaker = self._dominant_speaker(timeline)
        primary_speaker = "unknown"
        # if primary_speaker == "unknown":
        #     primary_speaker = self._last_final_speaker
        # else:
        #     self._last_final_speaker = primary_speaker

        delivery_chunks = self._split_for_delivery(full_text, self.max_words)
        if not delivery_chunks:
            delivery_chunks = [full_text]

        for sentence_index, chunk_text in enumerate(delivery_chunks):
            yield self._build_event(
                event_type="final",
                text=chunk_text,
                full_text=full_text,
                speaker=primary_speaker,
                # diarization={
                #     "status": "final",
                #     "speaker": primary_speaker,
                #     "segments": [
                #         {
                #             "start": round(start, 3),
                #             "end": round(end, 3),
                #             "speaker": speaker,
                #         }
                #         for start, end, speaker in timeline
                #     ],
                # },
                start_sample=start_sample,
                end_sample=end_sample,
                reason=reason,
                sentence_index=sentence_index,
                final=True,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_event(
        self,
        *,
        event_type: str,
        text: str,
        full_text: str,
        speaker: str,
        # diarization: dict,
        start_sample: int,
        end_sample: int,
        reason: str,
        sentence_index: int,
        final: bool,
    ) -> dict:
        self._event_index += 1
        start_sec = start_sample / SAMPLE_RATE
        end_sec = end_sample / SAMPLE_RATE
        return {
            "id": self._event_index,
            "segment_id": self._segment_index,
            "type": event_type,
            "final": final,
            "reason": reason,
            "text": text,
            "speaker": speaker,
            "sentence_index": sentence_index,
            "segment": {
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "duration": round(max(0.0, end_sec - start_sec), 3),
            },
            "asr": {
                "text": text,
                "full_text": full_text,
                "word_count": _word_count(full_text),
            },
            # "diarization": diarization,
            "debug": {
                "stream_samples": self._stream_samples,
                "speaking": self._speaking,
                "chunk_ms": round((end_sample - start_sample) * 1000 / SAMPLE_RATE, 1),
            },
        }

    @staticmethod
    def _split_for_delivery(text: str, max_words: int) -> list[str]:
        sentences = AudioBuffer.split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            if len(words) <= max_words:
                chunks.append(sentence.strip())
                continue

            for start in range(0, len(words), max_words):
                chunk = " ".join(words[start : start + max_words]).strip()
                if chunk:
                    chunks.append(chunk)

        return chunks

    # @staticmethod
    # def _parse_diarization(
    #     raw_segments: list[str],
    #     seg: np.ndarray,
    # ) -> list[tuple[float, float, str]]:
    #     timeline: list[tuple[float, float, str]] = []
    #     duration = len(seg) / SAMPLE_RATE
    #
    #     for line in raw_segments:
    #         parts = str(line).strip().split()
    #         if len(parts) < 3:
    #             continue
    #         try:
    #             start, end = float(parts[0]), float(parts[1])
    #         except ValueError:
    #             continue
    #         speaker = " ".join(parts[2:])
    #         start = max(0.0, start)
    #         end = min(duration, end)
    #         if end > start:
    #             timeline.append((start, end, speaker))
    #
    #     timeline.sort(key=lambda item: (item[0], item[1]))
    #     return timeline
    #
    # @staticmethod
    # def _dominant_speaker(timeline: list[tuple[float, float, str]]) -> str:
    #     if not timeline:
    #         return "unknown"
    #     return max(timeline, key=lambda item: item[1] - item[0])[2]

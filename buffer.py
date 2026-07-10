"""
Audio buffer with VAD-driven utterance detection and sentence-boundary splitting.

Design
------
A single accumulated buffer holds all incoming float32 PCM samples.
A ``_vad_pos`` pointer advances 512 samples at a time. When VAD detects
speech→silence transition (silence_gap elapsed), the speech segment is
sliced from the buffer and emitted as an utterance.

When ``finish()`` is called (stream end or timeout), whatever remains
from ``_speech_start`` to ``_vad_pos`` is flushed.

Methods
-------
feed(chunk)          — append raw PCM
check()              — advance VAD pointer, return completed utterances
finish()             — flush remaining speech as one utterance
split_sentences(txt) — static helper: split on terminal punctuation
"""

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # Silero VAD frame size

# Minimum silence (seconds) to close an utterance.
SILENCE_GAP = 0.6

# Terminal punctuation for sentence splitting.
SENTENCE_END = ".!?…。؟！।"


class AudioBuffer:
    def __init__(self, vad, threshold: float = 0.5):
        vad.threshold = threshold
        self.vad = vad

        # Accumulated audio (may be None when empty).
        self._buf: np.ndarray | None = None

        # VAD scan position (sample index within _buf).
        self._vad_pos: int = 0

        # State machine
        self._speaking: bool = False
        self._speech_start: int = 0   # sample index where current speech began
        self._silence_run: int = 0    # consecutive silence samples after speech

        # Timing: the stream-time (seconds) corresponding to index 0 of _buf.
        self._buf_base_time: float = 0.0

    # -- properties -----------------------------------------------------

    @property
    def duration(self) -> float:
        """Seconds of unprocessed audio (from _vad_pos to end of buffer)."""
        remaining = 0
        if self._buf is not None:
            remaining = max(0, len(self._buf) - self._vad_pos)
        return remaining / SAMPLE_RATE

    # -- helpers --------------------------------------------------------

    def _idx_to_time(self, idx: int) -> float:
        """Convert a buffer sample index to absolute stream time."""
        return self._buf_base_time + idx / SAMPLE_RATE

    # -- feeding --------------------------------------------------------

    def feed(self, chunk: np.ndarray) -> None:
        """Append a chunk of float32 mono audio @ 16 kHz."""
        chunk = chunk.astype(np.float32)
        if self._buf is None:
            self._buf = chunk.copy()
            self._buf_base_time = 0.0
        else:
            self._buf = np.concatenate([self._buf, chunk])
        # Advance the "end" time; _buf_base_time stays the same.

    # -- utterance detection --------------------------------------------

    def check(self) -> list[tuple[np.ndarray, float, float]]:
        """
        Advance the VAD pointer over unprocessed audio.

        Returns a list of ``(segment, start_sec, end_sec)`` tuples for
        utterances closed by a silence gap ≥ ``SILENCE_GAP``.
        """
        if self._buf is None or self._vad_pos + CHUNK_SAMPLES > len(self._buf):
            return []

        buf = self._buf
        utterances: list[tuple[np.ndarray, float, float]] = []

        i = self._vad_pos
        while i + CHUNK_SAMPLES <= len(buf):
            frame = buf[i:i + CHUNK_SAMPLES]
            is_speech = self.vad.is_speech(frame)

            if is_speech:
                if not self._speaking:
                    self._speaking = True
                    self._speech_start = i
                    self._silence_run = 0
                self._silence_run = 0
            else:
                if self._speaking:
                    self._silence_run += CHUNK_SAMPLES
                    if self._silence_run >= SILENCE_GAP * SAMPLE_RATE:
                        # Close utterance at the point silence began
                        # (trim the silence tail).
                        end_idx = i
                        seg = buf[self._speech_start:end_idx]
                        if len(seg) >= 256:
                            t0 = self._idx_to_time(self._speech_start)
                            t1 = self._idx_to_time(end_idx)
                            utterances.append((seg, t0, t1))
                        self._speaking = False
                        self._silence_run = 0

            i += CHUNK_SAMPLES

        self._vad_pos = i

        # Trim the processed prefix to save memory.
        self._maybe_trim()

        return utterances

    def _maybe_trim(self) -> None:
        """Drop processed buffer prefix, adjusting indices + base time."""
        if self._buf is None:
            return
        # Keep at least 1 second before vad_pos to preserve context.
        keep_from = max(0, self._vad_pos - SAMPLE_RATE)
        if keep_from < 1024:
            return  # not worth the copy
        self._buf = self._buf[keep_from:]
        self._buf_base_time += keep_from / SAMPLE_RATE
        self._vad_pos -= keep_from
        if self._speaking:
            self._speech_start -= keep_from

    # -- force flush ----------------------------------------------------

    def finish(self) -> tuple[np.ndarray, float, float] | None:
        """
        Flush whatever speech is still buffered.

        Called when the stream ends or the buffer has grown too large
        without a silence gap.
        """
        if self._buf is None or len(self._buf) == 0:
            return None

        # The utterance is from _speech_start to _vad_pos.
        end = self._vad_pos
        start = self._speech_start if self._speaking else 0
        seg = self._buf[start:end]

        # Discard and reset
        self._buf = None
        self._vad_pos = 0
        self._speaking = False
        self._silence_run = 0

        if len(seg) < 256:
            return None

        t0 = self._idx_to_time(start)
        t1 = self._idx_to_time(end)
        return (seg, t0, t1)

    # -- sentence splitting ---------------------------------------------

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """
        Split *text* on terminal punctuation.

        The last segment (without terminal punctuation) is still returned
        so nothing is silently dropped.
        """
        text = text.strip()
        if not text:
            return []

        sentences: list[str] = []
        buf: list[str] = []

        for ch in text:
            buf.append(ch)
            if ch in SENTENCE_END:
                sentences.append("".join(buf).strip())
                buf = []

        tail = "".join(buf).strip()
        if tail:
            sentences.append(tail)

        return sentences

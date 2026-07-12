from __future__ import annotations

import logging

import nemo.collections.asr as nemo_asr

_ASR_MODEL = None
_ASR_DISABLED = False

logger = logging.getLogger(__name__)


def _get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        _ASR_MODEL = nemo_asr.models.ASRModel.restore_from(
            r"/mnt/d/models/tts/gguf/parakeet-tdt-0.6b-v3.nemo"
        )
    return _ASR_MODEL


def nemo_transcribe(data):
    """Transcribe float32 audio array @ 16kHz, return text string."""
    global _ASR_DISABLED

    # 3200 samples @ 16kHz = 0.2 seconds.
    if _ASR_DISABLED or data is None or len(data) < 3200:
        return ""

    try:
        model = _get_asr_model()
        output = model.transcribe([data])
        if not output:
            return ""
        return getattr(output[0], "text", "") or ""
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        logger.exception("ASR transcription failed; disabling ASR for this process. %s", message)

        # CUDA illegal access often poisons the process; stop retrying ASR.
        if "cuda" in message.lower() or "illegal memory access" in message.lower():
            _ASR_DISABLED = True

        return ""














import nemo.collections.asr as nemo_asr

_ASR_MODEL = None


def _get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        _ASR_MODEL = nemo_asr.models.ASRModel.restore_from(
            r"d:/models/tts/gguf/parakeet-tdt-0.6b-v3.nemo"
        )
    return _ASR_MODEL


def nemo_transcribe(data):
    """Transcribe float32 audio array @ 16kHz, return text string."""
    model = _get_asr_model()
    output = model.transcribe([data])
    if not output:
        return ""
    return getattr(output[0], "text", "") or ""














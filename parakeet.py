from __future__ import annotations

import logging

import nemo.collections.asr as nemo_asr

_ASR_MODEL = None
_ASR_DISABLED = False

logger = logging.getLogger(__name__)


def _get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        # Try local path first, fall back to model name for auto-download
        model_path = r"/mnt/d/models/tts/gguf/parakeet-tdt-0.6b-v3.nemo"
        try:
            _ASR_MODEL = nemo_asr.models.ASRModel.restore_from(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}. Trying to download...")
            # Fall back to downloading the model
            _ASR_MODEL = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet_ctc_small")
        
    from omegaconf import open_dict

    decoding_cfg = _ASR_MODEL.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = False
    _ASR_MODEL.change_decoding_strategy(decoding_cfg)
    return _ASR_MODEL


def nemo_transcribe(data):
    """Transcribe float32 audio array @ 16kHz, return text string."""
    global _ASR_DISABLED

    # 3200 samples @ 16kHz = 0.2 seconds.
    if _ASR_DISABLED or data is None or len(data) < 3200:
        return ""

    try:
        model = _get_asr_model()
        output = model.transcribe([data], timestamps=True)
        word_timestamps = output[0].timestamp["word"]

        if not word_timestamps:
            return ""
        
        text = " ".join(w.get("word", "") for w in word_timestamps)
        return text
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        logger.exception("ASR transcription failed; disabling ASR for this process. %s", message)

        # CUDA illegal access often poisons the process; stop retrying ASR.
        if "cuda" in message.lower() or "illegal memory access" in message.lower():
            _ASR_DISABLED = True

        return ""


def nemo_transcribe_with_timestamps(data):
    """Transcribe float32 audio array @ 16kHz, return list of word dicts with timestamps."""
    global _ASR_DISABLED

    # 3200 samples @ 16kHz = 0.2 seconds.
    if _ASR_DISABLED or data is None or len(data) < 3200:
        return []

    try:
        model = _get_asr_model()
        output = model.transcribe([data], timestamps=True)
        word_timestamps = output[0].timestamp["word"]

        if not word_timestamps:
            return []
        
        return word_timestamps
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        logger.exception("ASR transcription failed; disabling ASR for this process. %s", message)

        # CUDA illegal access often poisons the process; stop retrying ASR.
        if "cuda" in message.lower() or "illegal memory access" in message.lower():
            _ASR_DISABLED = True

        return []














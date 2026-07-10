from nemo.collections.asr.models import SortformerEncLabelModel

_DIAR_MODEL = None


def _get_diar_model():
    global _DIAR_MODEL
    if _DIAR_MODEL is None:
        _DIAR_MODEL = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
        _DIAR_MODEL.eval()

        _DIAR_MODEL.sortformer_modules.chunk_len = 340
        _DIAR_MODEL.sortformer_modules.chunk_right_context = 40
        _DIAR_MODEL.sortformer_modules.fifo_len = 40
        _DIAR_MODEL.sortformer_modules.spkcache_update_period = 300
    return _DIAR_MODEL


def nemo_dia(audio_npy):
    model = _get_diar_model()
    predicted_segments = model.diarize(audio=audio_npy, batch_size=1, sample_rate=16000)

    # predicted_segments looks like this:
    # ['0.400 3.040 speaker_0', ...]
    return predicted_segments

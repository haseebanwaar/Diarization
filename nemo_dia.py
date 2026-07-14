import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel


SAMPLE_RATE = 16000
SPEAKER_ANCHOR_TAIL_SEC = 2.0
_DIAR_MODEL = None


# #for streaming
# diar_model.sortformer_modules.chunk_len = 6
# diar_model.sortformer_modules.chunk_right_context = 7
# diar_model.sortformer_modules.fifo_len = 188
# diar_model.sortformer_modules.spkcache_update_period = 144
#
# #for streaming
# diar_model.sortformer_modules.chunk_len = 3
# diar_model.sortformer_modules.chunk_right_context = 1
# diar_model.sortformer_modules.fifo_len = 188
# diar_model.sortformer_modules.spkcache_update_period = 144


def _get_diar_model():
    global _DIAR_MODEL
    if _DIAR_MODEL is None:
        _DIAR_MODEL = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
        _DIAR_MODEL.eval()

        _DIAR_MODEL.sortformer_modules.chunk_len = 6
        _DIAR_MODEL.sortformer_modules.chunk_right_context = 7
        _DIAR_MODEL.sortformer_modules.fifo_len = 188*5
        _DIAR_MODEL.sortformer_modules.spkcache_update_period = 144
    return _DIAR_MODEL


def _normalise_predictions(predicted_segments):
    """Return NeMo's one-item batch as a consistently sorted list of strings."""
    if not predicted_segments:
        return []
    if isinstance(predicted_segments[0], list):
        predicted_segments = predicted_segments[0]

    def start_time(line):
        try:
            return float(str(line).strip().split()[0])
        except (IndexError, ValueError):
            return float("inf")

    return sorted((str(line) for line in predicted_segments), key=start_time)


def _parse_segment(line):
    parts = str(line).strip().split()
    if len(parts) < 3:
        return None
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if end <= start:
        return None
    return start, end, " ".join(parts[2:]).strip()


def _previous_speaker_mapping(predicted_segments, previous_duration, stable_speaker):
    """Anchor this inference's local labels to the previous stable speaker label."""
    if previous_duration <= 0 or not stable_speaker or stable_speaker == "unknown":
        return {}

    durations = {}
    labels = set()
    anchor_start = max(0.0, previous_duration - SPEAKER_ANCHOR_TAIL_SEC)
    for line in predicted_segments:
        parsed = _parse_segment(line)
        if parsed is None:
            continue
        start, end, speaker = parsed
        labels.add(speaker)
        # The speaker at the end of the previous utterance is the useful
        # continuity anchor; a dominant speaker much earlier may be unrelated.
        overlap = max(0.0, min(end, previous_duration) - max(start, anchor_start))
        if overlap:
            durations[speaker] = durations.get(speaker, 0.0) + overlap

    if not durations:
        return {}

    local_previous_speaker = max(durations.items(), key=lambda item: item[1])[0]
    if local_previous_speaker == stable_speaker:
        return {}

    # Swap instead of replacing when the stable label is already present, so two
    # different local speakers never collapse into one output speaker.
    mapping = {local_previous_speaker: stable_speaker}
    if stable_speaker in labels:
        mapping[stable_speaker] = local_previous_speaker
    return mapping


def nemo_dia(audio_npy, context=None):
    """
    Diarize audio with optional surrounding context.
    
    Args:
        audio_npy: Current segment to diarize
        context: Optional dict with 'previous' and 'next' context:
            {
                "previous": {"audio": np.ndarray, "speaker": str},
                "next": {"audio": np.ndarray, "speaker": str},
            }
    
    Returns:
        List of diarization segments in format "start end speaker"
    """
    model = _get_diar_model()
    
    # If context is provided, concatenate surrounding audio for better context.
    if context:
        audio_to_process = audio_npy
        offset_samples = 0
        
        if context.get("previous") and context["previous"].get("audio") is not None:
            prev_audio = context["previous"]["audio"]
            audio_to_process = np.concatenate([prev_audio, audio_npy])
            offset_samples = len(prev_audio)
        
        # Add next context if available
        if context.get("next") and context["next"].get("audio") is not None:
            next_audio = context["next"]["audio"]
            audio_to_process = np.concatenate([audio_to_process, next_audio])
        
        predicted_segments = _normalise_predictions(
            model.diarize(audio=audio_to_process, batch_size=1, sample_rate=SAMPLE_RATE)
        )

        # Adjust timestamps to current segment's coordinate system
        current_duration_sec = len(audio_npy) / SAMPLE_RATE
        offset_sec = offset_samples / SAMPLE_RATE
        previous = context.get("previous") or {}
        speaker_mapping = _previous_speaker_mapping(
            predicted_segments,
            offset_sec,
            previous.get("speaker"),
        )

        filtered_segments = []
        for line in predicted_segments:
            parsed = _parse_segment(line)
            if parsed is None:
                continue
            start, end, speaker = parsed
            # Only include segments that overlap with current audio.
            if end > offset_sec and start < offset_sec + current_duration_sec:
                adjusted_start = max(0.0, start - offset_sec)
                adjusted_end = min(current_duration_sec, end - offset_sec)
                speaker = speaker_mapping.get(speaker, speaker)
                filtered_segments.append(f"{adjusted_start:.3f} {adjusted_end:.3f} {speaker}")

        # Returning the unfiltered context-relative timestamps here would assign
        # previous/next-context speech to words in the current segment.
        return filtered_segments
    else:
        # No context: process current segment only
        return _normalise_predictions(
            model.diarize(audio=audio_npy, batch_size=1, sample_rate=SAMPLE_RATE)
        )

import numpy as np
import torch


class VADWrapper:
    """Silero VAD wrapper — returns speech probability per 512-sample frame."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=True,
            force_reload=False,
        )

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check a single frame (512 samples @ 16 kHz) for speech.

        Parameters
        ----------
        audio : np.ndarray
            Float32 audio of exactly 512 samples.

        Returns
        -------
        bool — True if speech probability exceeds threshold.
        """
        if len(audio) == 0:
            return False

        tensor = torch.from_numpy(audio.astype(np.float32))
        if len(tensor) < 512:
            tensor = torch.nn.functional.pad(tensor, (0, 512 - len(tensor)))

        with torch.no_grad():
            prob = self.model(tensor, 16000).item()

        return prob > self.threshold

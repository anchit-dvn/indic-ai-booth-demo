"""Speech-to-Text using Meta OmniASR-CTC-300M — 348+ languages on AMD ROCm."""

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, AutoProcessor
from config import STT_MODEL, GPU_DEVICE, USE_GPU


class STTEngine:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = self._get_device()
        self._load()

    def _get_device(self):
        if not USE_GPU:
            return "cpu"
        if torch.cuda.is_available():
            return GPU_DEVICE
        return "cpu"

    def _load(self):
        print(f"[STT] Loading {STT_MODEL} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(STT_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            STT_MODEL
        ).to(self.device)
        self.model.eval()
        print("[STT] Model loaded.")

    def transcribe(self, audio, language="Hindi"):
        """
        Transcribe audio to text using OmniASR-CTC-300M.
        audio: (sample_rate, numpy_array) tuple from Gradio
        language: display name (used for logging; OmniASR is multilingual)
        Returns: transcribed text (str)
        """
        if audio is None:
            return "[No audio provided]"

        sample_rate, audio_data = audio

        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=16000,
            )
            sample_rate = 16000

        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.astype(np.float32)

        inputs = self.processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(pred_ids[0])
        return transcript.strip()

"""Language Identification using Facebook MMS-LID-1024 — detects spoken language from audio."""

import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from config import LID_MODEL, GPU_DEVICE, USE_GPU, LID_TO_LANG


class LIDEngine:
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
        print(f"[LID] Loading {LID_MODEL} on {self.device}...")
        self.processor = AutoFeatureExtractor.from_pretrained(LID_MODEL)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            LID_MODEL
        ).to(self.device)
        self.model.eval()
        print("[LID] Model loaded.")

    def identify(self, audio):
        """
        Identify the spoken language from audio.
        audio: (sample_rate, numpy_array) tuple from Gradio
        Returns: (detected_language_display_name, confidence_score)
        """
        if audio is None:
            return "Unknown", 0.0

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

        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_id = torch.max(probs, dim=-1)
        lid_code = self.model.config.id2label[predicted_id.item()]

        # Map ISO 639-3 code to our display name
        display_name = LID_TO_LANG.get(lid_code, lid_code)
        return display_name, confidence.item()

    def identify_top_k(self, audio, k=3):
        """
        Identify top-k candidate languages.
        Returns: list of (display_name, confidence) tuples
        """
        if audio is None:
            return [("Unknown", 0.0)]

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

        probs = torch.nn.functional.softmax(logits, dim=-1)
        topk_conf, topk_ids = torch.topk(probs, k=k, dim=-1)

        results = []
        for conf, lid_id in zip(topk_conf[0], topk_ids[0]):
            lid_code = self.model.config.id2label[lid_id.item()]
            display_name = LID_TO_LANG.get(lid_code, lid_code)
            results.append((display_name, conf.item()))

        return results

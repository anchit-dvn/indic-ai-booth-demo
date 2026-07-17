"""Text-to-Speech using Kokoro-82M via IndicVoice — espeak-ng G2P for Indic languages."""

import numpy as np
from config import LANGUAGES, TTS_MODEL


class TTSEngine:
    def __init__(self):
        self.pipelines = {}  # language → IndicPipeline instance
        self.sample_rate = 24000
        self._load()

    def _load(self):
        print("[TTS] IndicVoice (Kokoro-82M + espeak-ng G2P) ready for lazy loading.")

    def _get_indic_code(self, language):
        """Get the indic-voice language code for a display name."""
        entry = LANGUAGES.get(language)
        if entry is None:
            return None
        return entry[3]  # indic_voice_code

    def _load_pipeline(self, language):
        """Lazily load an IndicPipeline for the given language."""
        if language in self.pipelines:
            return self.pipelines[language]

        indic_code = self._get_indic_code(language)
        if indic_code is None:
            print(f"[TTS] {language} not supported by IndicVoice. Falling back to English.")
            indic_code = "en"
            language = "English"

        print(f"[TTS] Loading IndicPipeline for {language} (code={indic_code})...")
        try:
            from indicvoice import IndicPipeline
            pipeline = IndicPipeline(
                lang_code=indic_code,
                repo_id=TTS_MODEL,
            )
            self.pipelines[language] = pipeline
            print(f"[TTS] Loaded {language}.")
            return pipeline
        except ImportError:
            print("[TTS] indicvoice package not installed. Install with:")
            print("  pip install git+https://github.com/Bindkushal/indic-g2p.git")
            print("  pip install git+https://github.com/Bindkushal/indic-voice.git")
            print("  apt-get install espeak-ng")
            return None
        except Exception as e:
            print(f"[TTS] Failed to load {language}: {e}")
            return None

    def synthesize(self, text, language="Hindi", voice="af_heart"):
        """
        Synthesize text to speech using Kokoro-82M with Indic G2P.
        text: str in the target language
        language: display name from LANGUAGES dict
        voice: Kokoro voice name (af_heart, af_bella, am_adam, etc.)
        Returns: (sample_rate, numpy_array) tuple for Gradio audio output
        """
        if not text.strip():
            return (self.sample_rate, np.zeros(1000, dtype=np.float32))

        pipeline = self._load_pipeline(language)
        if pipeline is None:
            return (self.sample_rate, np.zeros(1000, dtype=np.float32))

        try:
            audio_chunks = []
            for gs, ps, audio in pipeline(text, voice=voice):
                if audio is not None:
                    audio_chunks.append(np.array(audio, dtype=np.float32))

            if not audio_chunks:
                return (self.sample_rate, np.zeros(1000, dtype=np.float32))

            waveform = np.concatenate(audio_chunks)
            return (self.sample_rate, waveform)
        except Exception as e:
            print(f"[TTS] Synthesis failed for {language}: {e}")
            return (self.sample_rate, np.zeros(1000, dtype=np.float32))

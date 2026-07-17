# Indic AI Booth Demo

Voice-first AI demo for Indic languages, running on AMD ROCm GPUs.
Built for the AMD AUP Learning Cloud university program booth.

## Quick Start (on AUP Learning Cloud)

1. **Launch a GPU notebook** on AUP Learning Cloud (any GPU resource)

2. **Open a terminal** in the notebook and clone this repo:
   ```bash
   git clone <your-repo-url> indic-ai-booth-demo
   cd indic-ai-booth-demo
   ```

3. **Install espeak-ng** (required for Kokoro TTS G2P):
   ```bash
   sudo apt-get install espeak-ng
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch the demo:**
   ```bash
   python app.py
   ```

6. Gradio will generate a **public URL** — open it in your browser.
   This is what you display on the booth screen.

## Demos

### Tab 1: "Speak to AI" (HERO DEMO)
Full voice-in → voice-out loop:
- Speak into the mic — language is auto-detected by MMS-LID-1024
- AI transcribes (OmniASR-CTC-300M) → extracts intent/sentiment (NLU) → generates response (Gemma 3 4B) → speaks back (Kokoro-82M + espeak-ng G2P)
- All on a single AMD GPU

### Tab 2: "Call Center AI"
- Select a scenario (banking, insurance, e-commerce, telecom)
- Upload or play a customer call recording
- AI transcribes, translates to English, analyzes sentiment, generates + speaks a response

### Tab 3: "Voice Translation"
- Speak in one language (e.g., English)
- AI translates and speaks the result in another (e.g., Hindi)
- Real-time voice-to-voice translation

### Tab 4: "Text Demos" (secondary)
- Indic language chatbot
- Text translation across Indic languages

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| LID | `facebook/mms-lid-1024` | Spoken language identification (1024 languages) |
| STT | `aadel4/omniASR-CTC-300M` | Speech-to-text (Meta OmniASR, 348+ languages) |
| LLM | `google/gemma-3-4b-it` | Response generation in Indic languages (Gemma 3 4B) |
| TTS | `Bindkushal/IndicVoice-82M` | Kokoro-82M TTS with native Indic G2P + espeak-ng fallback |
| Translation | `facebook/nllb-200-distilled-600M` | 200+ language translation |

All models run on AMD ROCm GPU (exposed as CUDA via HIP).

## Pre-Event Setup

1. **Pre-load models** — run `python app.py` once before the event to download all models
2. **Prepare audio samples** — place `.wav` files in `samples/` for call center demo
3. **Test mic input** — ensure the USB mic works with the Gradio interface
4. **Test speaker output** — ensure TTS audio plays through the booth speaker
5. **Verify GPU** — check the GPU status panel shows your AMD GPU

## Configuration

Edit `config.py` to change:
- Branding (name, tagline)
- Supported languages
- Model names
- Call center scenarios
- Server port

## File Structure

```
indic-ai-booth-demo/
├── app.py              # Main Gradio app (4 demo tabs)
├── config.py           # Configuration (branding, models, languages)
├── requirements.txt    # Python dependencies
├── modules/
│   ├── stt.py          # Speech-to-text (OmniASR-CTC-300M)
│   ├── lid.py          # Language identification (MMS-LID-1024)
│   ├── tts.py          # Text-to-speech (Kokoro-82M + espeak-ng G2P)
│   ├── nlu.py          # Intent + entity extraction
│   ├── llm.py          # Response generation (Gemma 3 4B IT)
│   ├── translation.py  # Translation (NLLB-200)
│   └── gpu_monitor.py  # AMD ROCm GPU monitoring
└── samples/            # Pre-recorded audio for call center demo
```

## Troubleshooting

- **No GPU detected:** Ensure ROCm is installed. Run `rocm-smi` to verify.
- **Model download fails:** Models download from HuggingFace. Ensure internet access.
- **TTS sounds robotic:** Kokoro-82M with Indic G2P produces natural voices for Hindi, Bengali, Punjabi, English. Other Indic languages fall back to English voice.
- **Mic not working:** Check browser permissions for microphone access.

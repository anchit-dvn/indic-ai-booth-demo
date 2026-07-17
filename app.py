"""
Indic AI Booth Demo — Main Gradio Application
Audio-first voice AI demos in Indic languages, running on AMD ROCm GPUs.

Launch:  python app.py
"""

import gradio as gr
import json
import time

from config import (
    BRAND_NAME, BRAND_TAGLINE, BRAND_DESCRIPTION,
    LANGUAGES, CALL_CENTER_SCENARIOS, SERVER_PORT, SHARE,
)
from modules.gpu_monitor import GPUMonitor

# ── Initialize engines (lazy loading to allow partial failures) ───────────────
_gpu = GPUMonitor()
_stt = None
_tts = None
_llm = None
_nlu = None
_translation = None
_lid = None


def _ensure_stt():
    global _stt
    if _stt is None:
        from modules.stt import STTEngine
        _stt = STTEngine()
    return _stt


def _ensure_tts():
    global _tts
    if _tts is None:
        from modules.tts import TTSEngine
        _tts = TTSEngine()
    return _tts


def _ensure_llm():
    global _llm
    if _llm is None:
        from modules.llm import LLMEngine
        _llm = LLMEngine()
    return _llm


def _ensure_nlu():
    global _nlu
    if _nlu is None:
        from modules.nlu import NLUEngine
        _nlu = NLUEngine(llm_engine=_llm)
    return _nlu


def _ensure_translation():
    global _translation
    if _translation is None:
        from modules.translation import TranslationEngine
        _translation = TranslationEngine()
    return _translation


def _ensure_lid():
    global _lid
    if _lid is None:
        from modules.lid import LIDEngine
        _lid = LIDEngine()
    return _lid


# ── Language dropdown options ─────────────────────────────────────────────────
LANG_CHOICES = list(LANGUAGES.keys())
LANG_CHOICES_WITH_AUTO = ["Auto-detect (MMS-LID)"] + LANG_CHOICES


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 1: "Speak to AI in Your Language" — Voice-in → Voice-out loop
# ═══════════════════════════════════════════════════════════════════════════════

def demo1_voice_loop(audio, language):
    """Full voice-in → LID → STT → NLU → LLM → TTS → voice-out pipeline."""
    if audio is None:
        return "Please record or upload audio.", "", "", "", "", None, _gpu.get_status_text()

    t0 = time.time()

    # Step 0: Language identification (if auto-detect) or use selected
    lid_text = ""
    if language == "Auto-detect (MMS-LID)":
        lid = _ensure_lid()
        detected_lang, confidence = lid.identify(audio)
        language = detected_lang
        lid_text = f"Detected: {detected_lang} (confidence: {confidence:.1%})"
        t_lid = time.time() - t0
    else:
        lid_text = f"Selected: {language}"
        t_lid = 0.0

    # Step 1: Speech-to-text (OmniASR-CTC-300M)
    stt = _ensure_stt()
    transcript = stt.transcribe(audio, language=language)
    t_stt = time.time() - t0

    if not transcript or transcript == "[No audio provided]":
        return "Could not transcribe audio. Please try again.", lid_text, "", "", "", None, _gpu.get_status_text()

    # Step 2: NLU — intent + entity extraction
    nlu = _ensure_nlu()
    nlu_result = nlu.extract(transcript, language=language)
    nlu_text = json.dumps(nlu_result, indent=2, ensure_ascii=False)

    # Step 3: LLM — generate response in the same language (Gemma 3 4B)
    llm = _ensure_llm()
    response = llm.generate_response(transcript, language=language)

    # Step 4: TTS — speak the response (Kokoro-82M + espeak-ng G2P)
    tts = _ensure_tts()
    response_audio = tts.synthesize(response, language=language)

    t_total = time.time() - t0
    timing = f"LID: {t_lid:.1f}s | STT: {t_stt:.1f}s | Total: {t_total:.1f}s"

    return transcript, lid_text, nlu_text, response, timing, response_audio, _gpu.get_status_text()


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 2: "Your Call Center, Automated"
# ═══════════════════════════════════════════════════════════════════════════════

def demo2_call_center(scenario_name, audio):
    """Process a customer call: transcribe → sentiment → summary → response."""
    scenario = CALL_CENTER_SCENARIOS.get(scenario_name, {})
    language = scenario.get("language", "Hindi")
    context = scenario.get("context", "")

    if audio is None:
        return "Please select a scenario and provide audio.", "", "", "", "", None, _gpu.get_status_text()

    t0 = time.time()

    # Transcribe
    stt = _ensure_stt()
    transcript = stt.transcribe(audio, language=language)

    # NLU
    nlu = _ensure_nlu()
    nlu_result = nlu.extract(transcript, language=language)
    sentiment = nlu_result.get("sentiment", "neutral")
    intent = nlu_result.get("intent", "unknown")
    summary = nlu_result.get("summary", "")

    # Translate transcript to English for the CIO
    translation = _ensure_translation()
    english_transcript = translation.translate(transcript, source_lang=language, target_lang="English")

    # Generate response
    llm = _ensure_llm()
    response = llm.generate_response(transcript, language=language, context=context)

    # Speak the response
    tts = _ensure_tts()
    response_audio = tts.synthesize(response, language=language)

    t_total = time.time() - t0

    return (
        transcript,
        english_transcript,
        f"Intent: {intent}\nSentiment: {sentiment}\nSummary: {summary}",
        response,
        f"Processed in {t_total:.1f}s",
        response_audio,
        _gpu.get_status_text(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 3: "Voice Translation — Real-Time"
# ═══════════════════════════════════════════════════════════════════════════════

def demo3_voice_translation(audio, source_lang, target_lang):
    """Voice-to-voice translation: LID → STT → translate → TTS."""
    if audio is None:
        return "Please record or upload audio.", "", "", "", None, _gpu.get_status_text()

    t0 = time.time()

    # Auto-detect source language if requested
    lid_text = ""
    if source_lang == "Auto-detect (MMS-LID)":
        lid = _ensure_lid()
        source_lang, confidence = lid.identify(audio)
        lid_text = f"Detected: {source_lang} (confidence: {confidence:.1%})"

    # STT in source language (OmniASR-CTC-300M)
    stt = _ensure_stt()
    transcript = stt.transcribe(audio, language=source_lang)

    # Translate (NLLB-200)
    translation = _ensure_translation()
    translated = translation.translate(transcript, source_lang=source_lang, target_lang=target_lang)

    # TTS in target language (Kokoro-82M + espeak-ng G2P)
    tts = _ensure_tts()
    translated_audio = tts.synthesize(translated, language=target_lang)

    t_total = time.time() - t0

    return lid_text, transcript, translated, f"Translated in {t_total:.1f}s", translated_audio, _gpu.get_status_text()


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 4: Text demos (secondary)
# ═══════════════════════════════════════════════════════════════════════════════

def demo4_chatbot(message, language):
    """Text chatbot in Indic languages."""
    if not message.strip():
        return "Please enter a message.", ""
    llm = _ensure_llm()
    response = llm.generate_response(message, language=language)
    return response, _gpu.get_status_text()


def demo4_translate_text(text, source_lang, target_lang):
    """Text translation."""
    if not text.strip():
        return "Please enter text to translate.", ""
    translation = _ensure_translation()
    result = translation.translate(text, source_lang=source_lang, target_lang=target_lang)
    return result, _gpu.get_status_text()


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU Status updater
# ═══════════════════════════════════════════════════════════════════════════════

def update_gpu_status():
    return _gpu.get_status_text()


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0e0;
}
.gradio-container .main {
    background: transparent;
}
#brand-header {
    text-align: center;
    padding: 20px 0 10px 0;
    border-bottom: 1px solid #333;
    margin-bottom: 20px;
}
#brand-header h1 {
    color: #ff6b35;
    font-size: 2.5em;
    margin: 0;
}
#brand-header p {
    color: #aaa;
    font-size: 1.1em;
    margin: 5px 0 0 0;
}
.gpu-monitor {
    background: #1a1a2e !important;
    border: 1px solid #ff6b35 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-family: monospace !important;
    font-size: 0.9em !important;
    color: #4ecca3 !important;
}
.tab-label {
    font-size: 1.1em !important;
    font-weight: bold !important;
}
"""

with gr.Blocks(
    title=f"{BRAND_NAME} — Indic Voice AI",
) as app:

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row():
        gr.HTML(f"""
        <div id="brand-header">
            <h1>{BRAND_NAME}</h1>
            <p>{BRAND_TAGLINE}</p>
            <p style="font-size: 0.9em; color: #666;">{BRAND_DESCRIPTION}</p>
        </div>
        """)

    # ── GPU Monitor (always visible) ──────────────────────────────────────────
    gpu_status = gr.Textbox(
        label="AMD ROCm GPU Status",
        value=_gpu.get_status_text(),
        every=5,
        elem_classes=["gpu-monitor"],
        interactive=False,
        lines=4,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 1: Voice Loop (HERO DEMO)
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Speak to AI", elem_classes=["tab-label"]):
            gr.Markdown("### Speak in your language — AI listens, understands, and responds in voice")
            gr.Markdown("*Full voice-in → voice-out loop: MMS-LID → OmniASR → Gemma 3 4B → Kokoro TTS*")

            with gr.Row():
                with gr.Column(scale=1):
                    lang_dd1 = gr.Dropdown(
                        choices=LANG_CHOICES_WITH_AUTO,
                        value="Auto-detect (MMS-LID)",
                        label="Language (auto-detect uses MMS-LID-1024)",
                    )
                    audio_in1 = gr.Audio(
                        label="Speak here (or upload audio)",
                        type="numpy",
                        sources=["microphone", "upload"],
                    )
                    btn1 = gr.Button("Process Voice", variant="primary", size="lg")

                with gr.Column(scale=2):
                    lid_out1 = gr.Textbox(label="Language Identification (MMS-LID)", lines=1)
                    transcript_out1 = gr.Textbox(label="Transcription (OmniASR-CTC-300M)", lines=3)
                    nlu_out1 = gr.Textbox(label="AI Understanding (Intent + Entities + Sentiment)", lines=6)
                    response_out1 = gr.Textbox(label="AI Response (Gemma 3 4B)", lines=4)
                    timing_out1 = gr.Textbox(label="Processing Time", lines=1)
                    audio_out1 = gr.Audio(label="AI Voice Response (Kokoro-82M)", autoplay=True)

            btn1.click(
                fn=demo1_voice_loop,
                inputs=[audio_in1, lang_dd1],
                outputs=[transcript_out1, lid_out1, nlu_out1, response_out1, timing_out1, audio_out1, gpu_status],
            )

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 2: Call Center AI
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Call Center AI", elem_classes=["tab-label"]):
            gr.Markdown("### Your call center, automated — in Indic languages")
            gr.Markdown("*Play a customer call → AI transcribes, analyzes sentiment, generates response*")

            with gr.Row():
                with gr.Column(scale=1):
                    scenario_dd = gr.Dropdown(
                        choices=list(CALL_CENTER_SCENARIOS.keys()),
                        value=list(CALL_CENTER_SCENARIOS.keys())[0],
                        label="Scenario",
                    )
                    audio_in2 = gr.Audio(
                        label="Customer call audio",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                    btn2 = gr.Button("Analyze Call", variant="primary", size="lg")

                with gr.Column(scale=2):
                    transcript_out2 = gr.Textbox(label="Transcription (Original Language)", lines=3)
                    english_out2 = gr.Textbox(label="English Translation", lines=3)
                    analysis_out2 = gr.Textbox(label="AI Analysis (Intent + Sentiment + Summary)", lines=4)
                    response_out2 = gr.Textbox(label="Suggested Response", lines=4)
                    timing_out2 = gr.Textbox(label="Processing Time", lines=1)
                    audio_out2 = gr.Audio(label="AI Voice Response", autoplay=True)

            btn2.click(
                fn=demo2_call_center,
                inputs=[scenario_dd, audio_in2],
                outputs=[transcript_out2, english_out2, analysis_out2, response_out2, timing_out2, audio_out2, gpu_status],
            )

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 3: Voice Translation
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Voice Translation", elem_classes=["tab-label"]):
            gr.Markdown("### Real-time voice-to-voice translation across Indic languages")
            gr.Markdown("*Speak in one language → AI translates and speaks in another*")

            with gr.Row():
                with gr.Column(scale=1):
                    src_lang_dd = gr.Dropdown(
                        choices=LANG_CHOICES_WITH_AUTO,
                        value="Auto-detect (MMS-LID)",
                        label="From (auto-detect uses MMS-LID)",
                    )
                    tgt_lang_dd = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="Hindi",
                        label="To (target language)",
                    )
                    audio_in3 = gr.Audio(
                        label="Speak here (or upload audio)",
                        type="numpy",
                        sources=["microphone", "upload"],
                    )
                    btn3 = gr.Button("Translate Voice", variant="primary", size="lg")

                with gr.Column(scale=2):
                    lid_out3 = gr.Textbox(label="Language Identification", lines=1)
                    transcript_out3 = gr.Textbox(label="Original (transcribed)", lines=3)
                    translated_out3 = gr.Textbox(label="Translated", lines=3)
                    timing_out3 = gr.Textbox(label="Processing Time", lines=1)
                    audio_out3 = gr.Audio(label="Translated Voice Output (Kokoro-82M)", autoplay=True)

            btn3.click(
                fn=demo3_voice_translation,
                inputs=[audio_in3, src_lang_dd, tgt_lang_dd],
                outputs=[lid_out3, transcript_out3, translated_out3, timing_out3, audio_out3, gpu_status],
            )

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 4: Text Demos (secondary)
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Text Demos", elem_classes=["tab-label"]):
            gr.Markdown("### Text-based demos (secondary — audio demos are the main attraction)")

            with gr.Row():
                # Chatbot
                with gr.Column():
                    gr.Markdown("#### Indic Language Chatbot")
                    lang_dd4 = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="Hindi",
                        label="Response Language",
                    )
                    chat_in = gr.Textbox(label="Your message", lines=2, placeholder="Type a query...")
                    chat_btn = gr.Button("Send", variant="primary")
                    chat_out = gr.Textbox(label="AI Response", lines=4)

                    chat_btn.click(
                        fn=demo4_chatbot,
                        inputs=[chat_in, lang_dd4],
                        outputs=[chat_out, gpu_status],
                    )

                # Text translation
                with gr.Column():
                    gr.Markdown("#### Text Translation")
                    src_lang_dd4 = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="English",
                        label="From",
                    )
                    tgt_lang_dd4 = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="Hindi",
                        label="To",
                    )
                    trans_in = gr.Textbox(label="Text to translate", lines=2, placeholder="Enter text...")
                    trans_btn = gr.Button("Translate", variant="primary")
                    trans_out = gr.Textbox(label="Translation", lines=4)

                    trans_btn.click(
                        fn=demo4_translate_text,
                        inputs=[trans_in, src_lang_dd4, tgt_lang_dd4],
                        outputs=[trans_out, gpu_status],
                    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="text-align: center; padding: 20px 0; border-top: 1px solid #333; margin-top: 20px;">
        <p style="color: #666; font-size: 0.85em;">
            {BRAND_NAME} · Powered by AMD ROCm GPUs · {BRAND_TAGLINE}
        </p>
    </div>
    """)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  {BRAND_NAME} — Indic Voice AI Booth Demo")
    print(f"  Running on: {_gpu.device_name}")
    print(f"{'='*60}\n")

    app.launch(
        server_port=SERVER_PORT,
        share=SHARE,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css=CUSTOM_CSS,
    )

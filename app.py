"""
Indic AI Booth Demo — Main Gradio Application
Audio-first voice AI demos in Indic languages, running on AMD ROCm GPUs.

Launch:  python app.py
"""

import gradio as gr
import json
import time
import traceback

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


LANG_CHOICES = list(LANGUAGES.keys())
LANG_CHOICES_WITH_AUTO = ["Auto-detect"] + LANG_CHOICES


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 1: Voice Loop — LID → STT → NLU → LLM → TTS
# ═══════════════════════════════════════════════════════════════════════════════

def demo1_voice_bot(audio, language, history, detail_visible):
    """Voice bot: audio in → chat conversation out."""
    if audio is None:
        return history, None, "Upload or record audio first.", _gpu.get_status_text()

    t0 = time.time()
    status = ""
    response_audio = None

    try:
        # LID
        if language == "Auto-detect":
            lid = _ensure_lid()
            detected, conf = lid.identify(audio)
            language = detected
            lid_label = f"{detected} ({conf:.0%})"
        else:
            lid_label = language
        t_lid = time.time() - t0

        # STT
        stt = _ensure_stt()
        transcript = stt.transcribe(audio, language=language)
        t_stt = time.time() - t0

        if not transcript or transcript == "[No audio provided]":
            status = "No speech detected."
            return history, None, status, _gpu.get_status_text()

        # NLU
        nlu = _ensure_nlu()
        nlu_result = nlu.extract(transcript, language=language)

        # LLM
        llm = _ensure_llm()
        response = llm.generate_response(transcript, language=language)

        # TTS
        tts = _ensure_tts()
        response_audio = tts.synthesize(response, language=language)

        t_total = time.time() - t0
        status = f"{lid_label} | LID {t_lid:.1f}s | STT {t_stt:.1f}s | Total {t_total:.1f}s"

        # Build chat messages
        user_msg = transcript
        bot_msg = response
        if detail_visible:
            intent = nlu_result.get("intent", "?")
            sentiment = nlu_result.get("sentiment", "?")
            bot_msg += f"\n\n_Intent: {intent} | Sentiment: {sentiment}_"

        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": bot_msg},
        ]

    except Exception as e:
        status = f"Error: {e}"
        traceback.print_exc()

    return history, response_audio, status, _gpu.get_status_text()


def demo1_clear():
    return [], None, "", _gpu.get_status_text()


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 2: Call Center AI
# ═══════════════════════════════════════════════════════════════════════════════

def demo2_call_center(scenario_name, audio):
    scenario = CALL_CENTER_SCENARIOS.get(scenario_name, {})
    language = scenario.get("language", "Hindi")
    context = scenario.get("context", "")

    if audio is None:
        return ("", "", "", "", "Upload or record audio first.", None, _gpu.get_status_text())

    t0 = time.time()
    r = {"transcript": "", "english": "", "analysis": "", "response": "", "timing": "", "audio": None}

    try:
        stt = _ensure_stt()
        r["transcript"] = stt.transcribe(audio, language=language)

        nlu = _ensure_nlu()
        nlu_result = nlu.extract(r["transcript"], language=language)
        r["analysis"] = f"Intent: {nlu_result.get('intent', '?')}\nSentiment: {nlu_result.get('sentiment', '?')}\nSummary: {nlu_result.get('summary', '')}"

        translation = _ensure_translation()
        r["english"] = translation.translate(r["transcript"], source_lang=language, target_lang="English")

        llm = _ensure_llm()
        r["response"] = llm.generate_response(r["transcript"], language=language, context=context)

        tts = _ensure_tts()
        r["audio"] = tts.synthesize(r["response"], language=language)

        r["timing"] = f"Processed in {time.time() - t0:.1f}s"
    except Exception as e:
        r["timing"] = f"Error: {e}"
        traceback.print_exc()

    return (r["transcript"], r["english"], r["analysis"], r["response"], r["timing"], r["audio"], _gpu.get_status_text())


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 3: Voice Translation
# ═══════════════════════════════════════════════════════════════════════════════

def demo3_voice_translation(audio, source_lang, target_lang):
    if audio is None:
        return ("", "", "", "Upload or record audio first.", None, _gpu.get_status_text())

    t0 = time.time()
    r = {"lid": "", "transcript": "", "translated": "", "timing": "", "audio": None}

    try:
        if source_lang == "Auto-detect":
            lid = _ensure_lid()
            source_lang, conf = lid.identify(audio)
            r["lid"] = f"{source_lang} ({conf:.0%})"
        else:
            r["lid"] = source_lang

        stt = _ensure_stt()
        r["transcript"] = stt.transcribe(audio, language=source_lang)

        translation = _ensure_translation()
        r["translated"] = translation.translate(r["transcript"], source_lang=source_lang, target_lang=target_lang)

        tts = _ensure_tts()
        r["audio"] = tts.synthesize(r["translated"], language=target_lang)

        r["timing"] = f"Translated in {time.time() - t0:.1f}s"
    except Exception as e:
        r["timing"] = f"Error: {e}"
        traceback.print_exc()

    return (r["lid"], r["transcript"], r["translated"], r["timing"], r["audio"], _gpu.get_status_text())


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO 4: Text demos
# ═══════════════════════════════════════════════════════════════════════════════

def demo4_chatbot(message, language):
    if not message.strip():
        return "Type a message first.", _gpu.get_status_text()
    try:
        llm = _ensure_llm()
        response = llm.generate_response(message, language=language)
        return response, _gpu.get_status_text()
    except Exception as e:
        return f"Error: {e}", _gpu.get_status_text()


def demo4_translate_text(text, source_lang, target_lang):
    if not text.strip():
        return "Enter text to translate.", _gpu.get_status_text()
    try:
        translation = _ensure_translation()
        result = translation.translate(text, source_lang=source_lang, target_lang=target_lang)
        return result, _gpu.get_status_text()
    except Exception as e:
        return f"Error: {e}", _gpu.get_status_text()


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI — dark, minimal, clean
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
:root { --bg:#0a0a0f; --surface:#14141f; --surface2:#1c1c2a; --border:#2a2a3a; --text:#e4e4e7; --text-dim:#71717a; --accent:#f97316; --accent-dim:#c2410c; --green:#22c55e; }
.gradio-container { background:var(--bg)!important; color:var(--text)!important; max-width:1200px!important; margin:0 auto!important; font-family:system-ui,-apple-system,sans-serif!important; }
.gradio-container .main { background:transparent!important; padding:0 12px!important; }
#header { text-align:center; padding:24px 0 16px; }
#header h1 { color:var(--text)!important; font-size:1.6em; font-weight:700; margin:0; }
#header .tagline { color:var(--accent)!important; font-size:0.9em; margin:4px 0 0; font-weight:500; }
#header .desc { color:var(--text-dim)!important; font-size:0.75em; margin:4px 0 0; }
#gpu-bar { background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:8px!important; padding:8px 14px!important; font-family:ui-monospace,monospace!important; font-size:0.75em!important; color:var(--green)!important; margin-bottom:16px!important; }
.gradio-container .tabs { border:none!important; }
.tab-nav { border-bottom:1px solid var(--border)!important; gap:4px!important; }
.tab-nav button { color:var(--text-dim)!important; font-size:0.85em!important; font-weight:500!important; padding:8px 18px!important; border:none!important; border-bottom:2px solid transparent!important; background:transparent!important; }
.tab-nav button.selected { color:var(--accent)!important; border-bottom:2px solid var(--accent)!important; }
.gradio-container .tabitem { padding-top:16px!important; }
.gradio-container input, .gradio-container textarea { background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:8px!important; color:var(--text)!important; font-size:0.9em!important; }
.gradio-container .dropdown { background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:8px!important; }
.gradio-container button.primary { background:var(--accent)!important; border:none!important; border-radius:8px!important; color:#fff!important; font-weight:600!important; width:100%!important; margin-top:8px!important; }
.gradio-container button.primary:hover { background:var(--accent-dim)!important; }
.gradio-container label { color:var(--text-dim)!important; font-size:0.75em!important; font-weight:500!important; margin-bottom:4px!important; }
.gradio-container .markdown { color:var(--text-dim)!important; font-size:0.8em!important; }
.gradio-container .markdown h3 { color:var(--text)!important; font-size:1em!important; font-weight:600!important; margin:0 0 12px!important; }
.gradio-container .output-text { background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:8px!important; }
.gradio-container .form { gap:12px!important; }
.gradio-container .row { gap:16px!important; }
.gradio-container .column { gap:10px!important; }
/* Audio player sizing */
.gradio-container .audio-container { height:140px!important; min-height:140px!important; }
.gradio-container .audio-container audio { border-radius:8px!important; }
#footer { text-align:center; padding:14px 0; border-top:1px solid var(--border); margin-top:24px; }
#footer p { color:var(--text-dim)!important; font-size:0.7em; margin:0; }
footer { display:none!important; }
.gradio-container .chatbot { background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:12px!important; }
.gradio-container .chatbot .message { border-radius:12px!important; font-size:0.9em!important; }
.gradio-container .chatbot .message.user { background:var(--surface2)!important; color:var(--text)!important; }
.gradio-container .chatbot .message.bot { background:#1a1205!important; color:var(--text)!important; border:1px solid var(--accent-dim)!important; }
"""

with gr.Blocks(title=f"{BRAND_NAME}") as app:

    gr.HTML(f"""
    <div id="header">
        <h1>{BRAND_NAME}</h1>
        <p class="tagline">{BRAND_TAGLINE}</p>
        <p class="desc">{BRAND_DESCRIPTION}</p>
    </div>
    """)

    gpu_status = gr.Textbox(
        value=_gpu.get_status_text(),
        every=5,
        elem_id="gpu-bar",
        interactive=False,
        show_label=False,
        lines=2,
    )

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 1: Voice AI
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Voice AI"):
            # Controls row
            with gr.Row():
                lang_dd1 = gr.Dropdown(
                    choices=LANG_CHOICES_WITH_AUTO,
                    value="Auto-detect",
                    label="Language",
                    scale=2,
                )
                detail_toggle = gr.Checkbox(
                    label="Show NLU details",
                    value=False,
                    scale=1,
                )
                with gr.Column(scale=1):
                    gr.Markdown("")
                    clear_btn1 = gr.Button("Clear chat", variant="secondary", size="sm")

            # Chat area with welcome message
            chatbot1 = gr.Chatbot(
                label="Conversation",
                height=380,
                show_label=False,
                value=[
                    {
                        "role": "assistant",
                        "content": "👋 Upload an audio file (WAV or MP3) and I'll transcribe, understand, and respond in voice. In the cloud, use upload; on a local laptop, you can also use the microphone."
                    }
                ],
            )

            # Audio input + Send
            with gr.Row():
                audio_in1 = gr.Audio(
                    label="Upload audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                    scale=5,
                )
                btn1 = gr.Button("Send", variant="primary", scale=1, size="lg")

            status_out1 = gr.Textbox(
                label="Status",
                lines=1,
                interactive=False,
                show_label=False,
                value="Ready. Upload audio and press Send.",
            )
            audio_out1 = gr.Audio(label="AI voice response", autoplay=True, visible=False)

            btn1.click(
                fn=demo1_voice_bot,
                inputs=[audio_in1, lang_dd1, chatbot1, detail_toggle],
                outputs=[chatbot1, audio_out1, status_out1, gpu_status],
            )
            clear_btn1.click(
                fn=demo1_clear,
                outputs=[chatbot1, audio_out1, status_out1, gpu_status],
            )

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 2: Call Center
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Call Center"):
            gr.Markdown("### Automated call analysis — transcribe, analyze, respond")

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    scenario_dd = gr.Dropdown(
                        choices=list(CALL_CENTER_SCENARIOS.keys()),
                        value=list(CALL_CENTER_SCENARIOS.keys())[0],
                        label="Scenario",
                    )
                    audio_in2 = gr.Audio(
                        label="Call audio",
                        type="numpy",
                        sources=["upload"],
                    )
                    btn2 = gr.Button("Analyze", variant="primary")

                with gr.Column(scale=1, min_width=500):
                    transcript_out2 = gr.Textbox(label="Transcription", lines=3, interactive=False)
                    english_out2 = gr.Textbox(label="English translation", lines=3, interactive=False)
                    analysis_out2 = gr.Textbox(label="Analysis", lines=3, interactive=False)
                    response_out2 = gr.Textbox(label="Suggested response", lines=4, interactive=False)
                    timing_out2 = gr.Textbox(label="Timing", lines=1, interactive=False)
                    audio_out2 = gr.Audio(label="Voice response", autoplay=True)

            btn2.click(
                fn=demo2_call_center,
                inputs=[scenario_dd, audio_in2],
                outputs=[transcript_out2, english_out2, analysis_out2, response_out2, timing_out2, audio_out2, gpu_status],
            )

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 3: Translation
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Translation"):
            gr.Markdown("### Voice-to-voice translation across Indic languages")

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    src_lang_dd = gr.Dropdown(
                        choices=LANG_CHOICES_WITH_AUTO,
                        value="Auto-detect",
                        label="From",
                    )
                    tgt_lang_dd = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="Hindi",
                        label="To",
                    )
                    audio_in3 = gr.Audio(
                        label="Audio input",
                        type="numpy",
                        sources=["upload"],
                    )
                    btn3 = gr.Button("Translate", variant="primary")

                with gr.Column(scale=1, min_width=500):
                    lid_out3 = gr.Textbox(label="Detected language", lines=1, interactive=False)
                    transcript_out3 = gr.Textbox(label="Original", lines=3, interactive=False)
                    translated_out3 = gr.Textbox(label="Translated", lines=3, interactive=False)
                    timing_out3 = gr.Textbox(label="Timing", lines=1, interactive=False)
                    audio_out3 = gr.Audio(label="Translated voice", autoplay=True)

            btn3.click(
                fn=demo3_voice_translation,
                inputs=[audio_in3, src_lang_dd, tgt_lang_dd],
                outputs=[lid_out3, transcript_out3, translated_out3, timing_out3, audio_out3, gpu_status],
            )

        # ═══════════════════════════════════════════════════════════════════════
        # TAB 4: Text
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("Text"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Chatbot")
                    lang_dd4 = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="Hindi",
                        label="Language",
                    )
                    chat_in = gr.Textbox(label="Message", lines=2, placeholder="Type a query...")
                    chat_btn = gr.Button("Send", variant="primary")
                    chat_out = gr.Textbox(label="Response", lines=4, interactive=False)

                    chat_btn.click(
                        fn=demo4_chatbot,
                        inputs=[chat_in, lang_dd4],
                        outputs=[chat_out, gpu_status],
                    )

                with gr.Column():
                    gr.Markdown("### Translate")
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
                    trans_in = gr.Textbox(label="Text", lines=2, placeholder="Enter text...")
                    trans_btn = gr.Button("Translate", variant="primary")
                    trans_out = gr.Textbox(label="Result", lines=4, interactive=False)

                    trans_btn.click(
                        fn=demo4_translate_text,
                        inputs=[trans_in, src_lang_dd4, tgt_lang_dd4],
                        outputs=[trans_out, gpu_status],
                    )

    gr.HTML(f"""
    <div id="footer">
        <p>{BRAND_NAME} | Powered by AMD ROCm | {BRAND_TAGLINE}</p>
    </div>
    """)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  {BRAND_NAME} — Indic Voice AI Booth Demo")
    print(f"  GPU: {_gpu.device_name}")
    print(f"{'='*60}\n")

    app.launch(
        server_port=SERVER_PORT,
        share=SHARE,
        show_error=True,
        css=CSS,
    )

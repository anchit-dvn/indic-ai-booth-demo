"""Configuration for Indic AI Booth Demo."""

# ── Branding ──────────────────────────────────────────────────────────────────
BRAND_NAME = "Devnagri AI"
BRAND_TAGLINE = "Enterprise AI in 20+ Indic Languages"
BRAND_DESCRIPTION = (
    "Full-stack voice & language AI for Hindi, Marathi, Tamil, Bengali, "
    "and more — running live on AMD ROCm GPUs."
)

# ── Supported Languages ───────────────────────────────────────────────────────
# Maps display name → (iso_code, nllb_code, mms_lid_iso639_3, indic_voice_code)
# indic_voice_code: language code for Kokoro/indic-voice TTS (None = not yet supported)
LANGUAGES = {
    "Hindi":    ("hi", "hin_Deva", "hin", "hi"),
    "Marathi":  ("mr", "mar_Deva", "mar", None),
    "Bengali":  ("bn", "ben_Beng", "ben", "bn"),
    "Tamil":    ("ta", "tam_Taml", "tam", None),
    "Telugu":   ("te", "tel_Telu", "tel", None),
    "Gujarati": ("gu", "guj_Gujr", "guj", None),
    "Kannada":  ("kn", "kan_Knda", "kan", None),
    "Malayalam":("ml", "mal_Mlym", "mal", None),
    "Punjabi":  ("pa", "pan_Guru", "pan", "pa"),
    "Urdu":     ("ur", "urd_Arab", "urd", None),
    "English":  ("en", "eng_Latn", "eng", "en"),
}

# Reverse map: MMS LID ISO 639-3 code → display name
LID_TO_LANG = {v[2]: k for k, v in LANGUAGES.items()}

# ── Model Configuration ──────────────────────────────────────────────────────
STT_MODEL = "aadel4/omniASR-CTC-300M"           # Speech-to-text (Meta OmniASR)
LID_MODEL = "facebook/mms-lid-1024"              # Language identification from speech
TTS_MODEL = "Bindkushal/IndicVoice-82M"          # Kokoro-82M TTS with Indic G2P
NLLB_MODEL = "facebook/nllb-200-distilled-600M"  # Translation
LLM_MODEL = "google/gemma-3-4b-it"               # Response generation (Gemma 3 4B)

# ── GPU / ROCm ────────────────────────────────────────────────────────────────
USE_GPU = True          # Auto-detect; falls back to CPU if no GPU
GPU_DEVICE = "cuda:0"   # ROCm exposes as CUDA via HIP

# ── Demo Scenarios ────────────────────────────────────────────────────────────
CALL_CENTER_SCENARIOS = {
    "Banking — Loan Inquiry (Hindi)": {
        "audio_file": "samples/hindi_banking_loan.wav",
        "language": "Hindi",
        "context": "Customer is asking about home loan eligibility and interest rates.",
    },
    "Insurance — Claim Status (Hindi)": {
        "audio_file": "samples/hindi_insurance_claim.wav",
        "language": "Hindi",
        "context": "Customer is asking about the status of their insurance claim.",
    },
    "E-commerce — Return Request (Tamil)": {
        "audio_file": "samples/tamil_ecommerce_return.wav",
        "language": "Tamil",
        "context": "Customer wants to return a defective product.",
    },
    "Telecom — Bill Dispute (Bengali)": {
        "audio_file": "samples/bengali_telecom_bill.wav",
        "language": "Bengali",
        "context": "Customer is disputing charges on their mobile bill.",
    },
}

# ── Server ────────────────────────────────────────────────────────────────────
SERVER_PORT = 7860
SHARE = True  # Gradio public link (useful for AUP Learning Cloud)

"""Translation using Facebook NLLB-200 — supports 200+ languages including Indic."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import NLLB_MODEL, LANGUAGES, GPU_DEVICE, USE_GPU


class TranslationEngine:
    def __init__(self):
        self.tokenizer = None
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
        print(f"[Translation] Loading {NLLB_MODEL} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[Translation] Model loaded.")

    def translate(self, text, source_lang="Hindi", target_lang="English"):
        """
        Translate text from source_lang to target_lang.
        Uses display names from LANGUAGES dict.
        """
        src_code = LANGUAGES.get(source_lang, ("", "hin_Deva"))[1]
        tgt_code = LANGUAGES.get(target_lang, ("", "eng_Latn"))[1]

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        forced_bos_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=200,
            )

        result = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )[0]
        return result.strip()

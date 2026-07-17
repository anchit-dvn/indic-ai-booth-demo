"""LLM — Response generation in Indic languages using Gemma 3 4B IT on AMD ROCm."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLM_MODEL, GPU_DEVICE, USE_GPU


class LLMEngine:
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
        print(f"[LLM] Loading {LLM_MODEL} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=dtype,
            device_map="auto" if self.device != "cpu" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        print("[LLM] Model loaded.")

    def is_available(self):
        return self.model is not None

    def generate(self, prompt, language="Hindi", max_tokens=256, system_prompt=None):
        """
        Generate a response using Gemma 3 4B IT.
        If language is not English, the system prompt instructs the model
        to respond in that language.
        """
        if system_prompt is None:
            system_prompt = (
                f"You are a helpful customer service assistant. "
                f"Respond in {language}. Be concise, professional, and helpful."
            )

        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{prompt}"},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def generate_response(self, user_text, language="Hindi", context=None):
        """
        Generate a customer service response to the user's query.
        context: optional business context (e.g., "banking", "insurance")
        """
        context_str = f"\nContext: {context}" if context else ""
        prompt = f"Customer query: {user_text}{context_str}\n\nProvide a helpful response."
        return self.generate(prompt, language=language)

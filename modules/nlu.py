"""NLU — Intent & Entity Extraction for Indic language inputs.

Uses the LLM to extract structured intent and entities from transcribed text.
Falls back to keyword matching if LLM is unavailable.
"""

import json
import re


INTENT_KEYWORDS = {
    "loan_inquiry": ["loan", "कर्ज", "ऋण", "karja", "கடன", "ঋণ"],
    "bill_dispute": ["bill", "बिल", "invoice", "चार्ज", "பில்", "বিল"],
    "return_request": ["return", "वापस", "वापसी", "திருப்பி", "ফেরত"],
    "claim_status": ["claim", "दावा", "உரிமைக்கோரிக்கை", "দাবি"],
    "account_inquiry": ["account", "खाता", "கணக்கு", "হিসাব"],
    "complaint": ["complaint", "शिकायत", "புகார்", "অভিযোগ"],
    "product_info": ["product", "उत्पाद", "தயாரிப்பு", "পণ্য"],
    "support": ["help", "मदद", "உதவி", "সাহায্য"],
}

ENTITY_PATTERNS = {
    "amount": r'(?:₹|Rs\.?|रु|₹)\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*(?:rupees|रुपये)',
    "phone": r'[\+]?[\d]{10,13}',
    "date": r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)',
    "order_id": r'(?:order|आर्डर|ஆர்டர்)\s*(?:id|number|#)?\s*[: ]?\s*([A-Z0-9]{6,})',
}


class NLUEngine:
    def __init__(self, llm_engine=None):
        self.llm = llm_engine

    def extract(self, text, language="Hindi"):
        """
        Extract intent and entities from text.
        Returns dict: {intent, entities, sentiment, summary}
        """
        if self.llm and self.llm.is_available():
            return self._extract_with_llm(text, language)
        return self._extract_with_keywords(text, language)

    def _extract_with_llm(self, text, language):
        prompt = f"""Analyze this {language} customer query and extract structured information.
Return ONLY valid JSON with these keys:
- intent: the customer's primary intent (in English)
- entities: dict of named entities found (amounts, dates, IDs, etc.)
- sentiment: one of "positive", "neutral", "frustrated", "angry"
- summary: one-line summary in English

Customer query ({language}): {text}

JSON:"""
        response = self.llm.generate(prompt, language="English", max_tokens=200)
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return self._extract_with_keywords(text, language)

    def _extract_with_keywords(self, text, language):
        text_lower = text.lower()

        intent = "general_inquiry"
        best_score = 0
        for intent_name, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                intent = intent_name

        entities = {}
        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if len(matches[0]) == 1 else matches[0]

        negative_words = ["नहीं", "बुरा", "இல்லை", "খারাপ", "problem", "issue", "wrong", "bad"]
        positive_words = ["अच्छा", "धन्यवाद", "நன்றி", "ধন্যবাদ", "good", "thank", "great"]

        neg_count = sum(1 for w in negative_words if w.lower() in text_lower)
        pos_count = sum(1 for w in positive_words if w.lower() in text_lower)

        if neg_count > pos_count:
            sentiment = "frustrated" if neg_count > 2 else "neutral"
        elif pos_count > neg_count:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        return {
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "summary": f"Customer query about {intent.replace('_', ' ')}",
        }

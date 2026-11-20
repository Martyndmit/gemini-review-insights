"""Словниковий метод — точка порівняння для Gemini і офлайн-режим.

Це навмисно простий підхід: тональність = кількість «позитивних» основ
мінус кількість «негативних», аспекти — за ключовими словами. Він не бачить
сарказму, заперечень і контексту, і саме це добре видно в порівнянні з LLM.
"""
from __future__ import annotations

import re
import time

from .analyzer import RunResult

# Загальні оцінні слова (основи). Словник складено без підгонки під демонабір:
# це типовий «перший підхід» аналітика, а не налаштована модель.
POSITIVE = [
    "дякую", "рекоменд", "чудов", "відмінн", "супер", "задовол", "зручн", "швидк",
    "приємн", "хорош", "якісн", "ввічлив", "вчасно", "найкращ", "great", "fast",
    "good", "excellent", "recommend", "thank", "love", "happy",
]
NEGATIVE = [
    "погано", "поган", "жахлив", "не рекоменд", "обман", "брак", "злама", "повільн",
    "груб", "відмов", "проблем", "скарг", "не працює", "не відповіда", "розчаров",
    "дорог", "bad", "slow", "broken", "terrible", "worst", "never", "problem",
]
# Ключові слова аспектів — загальна лексика інтернет-торгівлі.
ASPECTS = {
    "delivery": ["достав", "кур'єр", "посилк", "самовивіз", "відділенн", "delivery",
                 "shipping"],
    "product_quality": ["якіс", "брак", "злама", "не працює", "дефект", "quality",
                        "broken"],
    "price": ["цін", "дорог", "дешев", "знижк", "кешбек", "частинами", "price"],
    "customer_support": ["підтримк", "консультант", "гаряч", "оператор", "персонал",
                         "support"],
    "website_app": ["сайт", "застосун", "додат", "кошик", "website", "app "],
    "packaging": ["пакуванн", "упаков", "коробк", "packed", "packaging"],
    "returns_warranty": ["поверненн", "поверну", "гаранті", "ремонт", "обмін",
                         "warranty", "refund"],
}


def _count(text: str, stems: list[str]) -> int:
    return sum(1 for s in stems if s in text)


def analyze_one(review: dict) -> dict:
    text = review["text"].lower()
    pos, neg = _count(text, POSITIVE), _count(text, NEGATIVE)
    if pos and neg:
        sentiment = "mixed"
    elif pos:
        sentiment = "positive"
    elif neg:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    rating = {"positive": 5, "mixed": 3, "neutral": 3, "negative": 1}[sentiment]
    aspects = []
    for aspect, stems in ASPECTS.items():
        hit = next((s for s in stems if s in text), None)
        if hit:
            aspects.append({"aspect": aspect, "sentiment": sentiment if sentiment != "mixed"
                            else "neutral", "evidence": hit})
    is_question = "?" in text
    return {
        "review_id": review["id"],
        "language": "en" if re.fullmatch(r"[\x00-\x7f]+", review["text"]) else "uk",
        "sentiment": sentiment,
        "rating_estimate": rating,
        "aspects": aspects,
        "issues": [],
        "needs_reply": sentiment in ("negative", "mixed") or is_question,
        "summary_uk": "",
    }


def analyze(reviews: list[dict]) -> RunResult:
    t0 = time.perf_counter()
    run = RunResult(method="baseline", model=None)
    for r in reviews:
        run.analyses[r["id"]] = analyze_one(r)
    run.seconds = time.perf_counter() - t0
    return run

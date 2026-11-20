"""Структура відповіді, яку Gemini має повернути для кожного відгуку.

Тут дві речі, і вони навмисно різні:

1. response_schema() — КОМПАКТНА JSON Schema для параметра responseJsonSchema.
   API перетворює схему на скінченний автомат, що обмежує генерацію, і відхиляє
   занадто складні схеми (400 INVALID_ARGUMENT: "too many states for serving").
   Найбільше станів дають довгі описи, обмеження довжини масивів (minItems /
   maxItems, особливо вкладених) і числові діапазони (minimum / maximum).
   Тому в схемі лишено тільки структуру, типи й короткі enum.

2. Pydantic-моделі — повна перевірка на клієнті: діапазон оцінки, відомі
   аспекти, дублікати й пропуски. Кожен відгук перевіряється окремо, тож одна
   помилка моделі не знецінює всю порцію.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    mixed = "mixed"


class Aspect(str, Enum):
    delivery = "delivery"
    product_quality = "product_quality"
    price = "price"
    customer_support = "customer_support"
    website_app = "website_app"
    packaging = "packaging"
    returns_warranty = "returns_warranty"


SENTIMENT_UK = {
    "positive": "позитивна",
    "neutral": "нейтральна",
    "negative": "негативна",
    "mixed": "змішана",
}

ASPECT_UK = {
    "delivery": "доставка",
    "product_quality": "якість товару",
    "price": "ціна",
    "customer_support": "підтримка",
    "website_app": "сайт і застосунок",
    "packaging": "пакування",
    "returns_warranty": "повернення і гарантія",
}

MAX_ISSUES = 3


class AspectMention(BaseModel):
    aspect: Aspect
    sentiment: Sentiment
    evidence: str = ""


class ReviewAnalysis(BaseModel):
    """Результат для одного відгуку; правила заповнення — у системній інструкції."""
    review_id: str
    language: str = Field(default="")
    sentiment: Sentiment
    rating_estimate: int = Field(ge=1, le=5)
    aspects: list[AspectMention] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    needs_reply: bool
    summary_uk: str = ""

    @field_validator("issues")
    @classmethod
    def _limit_issues(cls, v: list[str]) -> list[str]:
        # стилістичне обмеження: зайві проблеми відкидаємо, а не бракуємо відгук
        return [s.strip() for s in v if s and s.strip()][:MAX_ISSUES]

    @field_validator("aspects")
    @classmethod
    def _dedupe_aspects(cls, v: list[AspectMention]) -> list[AspectMention]:
        seen, out = set(), []
        for m in v:
            if m.aspect not in seen:
                seen.add(m.aspect)
                out.append(m)
        return out

    @field_validator("language")
    @classmethod
    def _norm_language(cls, v: str) -> str:
        return v.strip().lower()[:5]


def _enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": values}


def response_schema() -> dict[str, Any]:
    """Компактна JSON Schema відповіді: лише структура, типи й короткі enum."""
    sentiments = [s.value for s in Sentiment]
    aspect = {
        "type": "object",
        "properties": {
            "aspect": _enum([a.value for a in Aspect]),
            "sentiment": _enum(sentiments),
            "evidence": {"type": "string"},
        },
        "required": ["aspect", "sentiment", "evidence"],
    }
    item = {
        "type": "object",
        "properties": {
            "review_id": {"type": "string"},
            "language": {"type": "string"},
            "sentiment": _enum(sentiments),
            "rating_estimate": {"type": "integer"},
            "aspects": {"type": "array", "items": aspect},
            "issues": {"type": "array", "items": {"type": "string"}},
            "needs_reply": {"type": "boolean"},
            "summary_uk": {"type": "string"},
        },
        "required": ["review_id", "language", "sentiment", "rating_estimate", "aspects",
                     "issues", "needs_reply", "summary_uk"],
    }
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": item}},
        "required": ["results"],
    }

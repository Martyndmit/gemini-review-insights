"""Ціни Gemini API (платний рівень, Standard), USD за 1 млн токенів.

Джерело: https://ai.google.dev/gemini-api/docs/pricing, перевірено 12.11.2026.
На безкоштовному рівні (free tier) ці моделі коштують $0; оцінка показує,
скільки той самий запуск коштував би на платному рівні.
Ціна виходу включає токени міркувань (thinking).
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRICES_CHECKED = "2026-11-12"

# model -> (input, output); для 3.8/3.6 Flash — вступні ціни до 31.12.2026,
# з 01.01.2027 вони стають $1.50 / $7.50.
PRICES_PER_1M = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


@dataclass
class Usage:
    """Сумарне споживання токенів за запуск (з usageMetadata відповідей)."""
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    per_request: list[dict] = field(default_factory=list)

    def add(self, meta, chunk_size: int, seconds: float) -> None:
        prompt = getattr(meta, "prompt_token_count", None) or 0
        out = getattr(meta, "candidates_token_count", None) or 0
        think = getattr(meta, "thoughts_token_count", None) or 0
        cached = getattr(meta, "cached_content_token_count", None) or 0
        self.requests += 1
        self.prompt_tokens += prompt
        self.output_tokens += out
        self.thinking_tokens += think
        self.cached_tokens += cached
        self.per_request.append({"reviews": chunk_size, "prompt_tokens": prompt,
                                 "output_tokens": out, "thinking_tokens": think,
                                 "seconds": round(seconds, 2)})

    def cost_usd(self, model: str) -> float | None:
        price = PRICES_PER_1M.get(model)
        if price is None:
            return None
        p_in, p_out = price
        return (self.prompt_tokens * p_in
                + (self.output_tokens + self.thinking_tokens) * p_out) / 1_000_000

    def as_dict(self, model: str) -> dict:
        cost = self.cost_usd(model)
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd_paid_tier": None if cost is None else round(cost, 6),
            "prices_checked": PRICES_CHECKED,
            "per_request": self.per_request,
        }

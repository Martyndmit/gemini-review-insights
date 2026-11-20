"""Тестовий стенд: справжній клієнт google-genai, але HTTP-відповіді імітуються.

Так перевіряється весь шлях через SDK — URL, заголовки, тіло запиту, розбір
відповіді та повтори — без мережі й без ключа. Тексти відповідей нижче
синтетичні: це не відповіді моделі, а фікстури у форматі REST API.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import httpx2 as httpx          # google-genai >= 3 працює на httpx2
except ImportError:                 # pragma: no cover
    import httpx

from review_insights.analyzer import make_client  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "reviews_demo.csv")


def ids_from_request(request) -> list[str]:
    body = json.loads(request.content)
    text = body["contents"][0]["parts"][0]["text"]
    payload = json.loads(text.split("\n", 1)[1])
    return [x["review_id"] for x in payload]


def fake_item(review_id: str, sentiment: str = "negative") -> dict:
    return {"review_id": review_id, "language": "uk", "sentiment": sentiment,
            "rating_estimate": 2, "aspects": [{"aspect": "delivery", "sentiment": "negative",
                                               "evidence": "доставку перенесли"}],
            "issues": ["затримка доставки"], "needs_reply": True,
            "summary_uk": "Клієнт скаржиться на доставку."}


def gemini_response(items: list[dict] | None = None, text: str | None = None,
                    finish: str = "STOP", prompt=100, out=50, thoughts=20) -> dict:
    if text is None:
        text = json.dumps({"results": items}, ensure_ascii=False)
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": finish, "index": 0}],
            "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": out,
                              "thoughtsTokenCount": thoughts,
                              "totalTokenCount": prompt + out + thoughts},
            "modelVersion": "gemini-3.8-flash"}


class FakeServer:
    """Обробник для httpx.MockTransport; behaviour(request, n) повертає httpx.Response."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.behaviour(request, len(self.requests))


@pytest.fixture
def make_fake_client():
    def factory(behaviour):
        server = FakeServer(behaviour)
        http = httpx.Client(transport=httpx.MockTransport(server))
        client = make_client(api_key="TEST-KEY", httpx_client=http, initial_delay=0.01)
        return client, server
    return factory


@pytest.fixture
def ok_behaviour():
    def behaviour(request, n):
        if request.url.path.endswith(":countTokens"):
            return httpx.Response(200, json={"totalTokens": 321})
        return httpx.Response(200, json=gemini_response(
            [fake_item(i) for i in ids_from_request(request)]))
    return behaviour


@pytest.fixture
def reviews():
    from review_insights.cli import read_reviews
    return read_reviews(DATA)

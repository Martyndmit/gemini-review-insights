import json

from conftest import fake_item, gemini_response, httpx, ids_from_request
from review_insights.analyzer import GeminiAnalyzer
from review_insights.schema import response_schema


def make(client, **kw):
    kw.setdefault("rpm", 0)
    return GeminiAnalyzer(client=client, log=lambda *_: None, **kw)


def test_request_shape(make_fake_client, ok_behaviour, reviews):
    client, server = make_fake_client(ok_behaviour)
    run = make(client, chunk_size=20).analyze(reviews[:3])
    req = server.requests[0]
    assert req.method == "POST"
    assert req.url.path.endswith("/models/gemini-3.8-flash:generateContent")
    assert req.headers["x-goog-api-key"] == "TEST-KEY"
    body = json.loads(req.content)
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    schema = gen["responseJsonSchema"]
    item = schema["properties"]["results"]["items"]
    assert item["properties"]["sentiment"]["enum"] == ["positive", "neutral", "negative", "mixed"]
    assert "ігноруй" in body["systemInstruction"]["parts"][0]["text"].lower()
    assert json.dumps(gen.get("thinkingConfig", {})).count("LOW") == 1
    assert set(run.analyses) == {"r01", "r02", "r03"}
    assert not run.failures and run.aborted is None


def test_chunking_and_usage(make_fake_client, ok_behaviour, reviews):
    client, server = make_fake_client(ok_behaviour)
    run = make(client, chunk_size=15).analyze(reviews)       # 40 -> 15 + 15 + 10
    assert len(server.requests) == 3
    assert [len(ids_from_request(r)) for r in server.requests] == [15, 15, 10]
    assert len(run.analyses) == 40
    u = run.usage.as_dict("gemini-3.8-flash")
    assert u["requests"] == 3 and u["prompt_tokens"] == 300 and u["thinking_tokens"] == 60
    # 300 вхідних * 0.75 + (150 вихідних + 60 thinking) * 3.75, за 1 млн
    assert abs(run.usage.cost_usd("gemini-3.8-flash") - (300 * 0.75 + 210 * 3.75) / 1e6) < 1e-12


def test_missing_review_is_retried(make_fake_client, reviews):
    def behaviour(request, n):
        ids = ids_from_request(request)
        if n == 1:   # модель «загубила» останній відгук порції
            return httpx.Response(200, json=gemini_response([fake_item(i) for i in ids[:-1]]))
        return httpx.Response(200, json=gemini_response([fake_item(i) for i in ids]))
    client, server = make_fake_client(behaviour)
    run = make(client, chunk_size=4).analyze(reviews[:4])
    assert len(server.requests) == 2
    assert ids_from_request(server.requests[1]) == ["r04"]
    assert set(run.analyses) == {"r01", "r02", "r03", "r04"}


def test_invalid_json_then_gives_up(make_fake_client, reviews):
    def behaviour(request, n):
        return httpx.Response(200, json=gemini_response(text='{"results": [', finish="MAX_TOKENS"))
    client, server = make_fake_client(behaviour)
    run = make(client, chunk_size=2, max_attempts=2).analyze(reviews[:2])
    # спроба 1: порція з 2; спроба 2: по одному відгуку
    assert len(server.requests) == 3
    assert run.analyses == {}
    assert set(run.failures) == {"r01", "r02"}
    assert "MAX_TOKENS" in run.failures["r01"]


def test_rate_limit_is_retried_by_sdk(make_fake_client, reviews):
    def behaviour(request, n):
        if n == 1:
            return httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                       "message": "Quota exceeded"}})
        return httpx.Response(200, json=gemini_response(
            [fake_item(i) for i in ids_from_request(request)]))
    client, server = make_fake_client(behaviour)
    run = make(client).analyze(reviews[:2])
    assert len(server.requests) == 2
    assert len(run.analyses) == 2


def test_client_error_aborts_and_keeps_progress(make_fake_client, reviews):
    def behaviour(request, n):
        if n == 2:
            return httpx.Response(403, json={"error": {"code": 403, "status": "PERMISSION_DENIED",
                                                       "message": "API key not valid"}})
        return httpx.Response(200, json=gemini_response(
            [fake_item(i) for i in ids_from_request(request)]))
    client, server = make_fake_client(behaviour)
    run = make(client, chunk_size=10).analyze(reviews[:30])
    assert len(run.analyses) == 10
    assert len(run.failures) == 20
    assert run.aborted.startswith("403")


def test_count_tokens(make_fake_client, ok_behaviour, reviews):
    client, server = make_fake_client(ok_behaviour)
    total = make(client, chunk_size=20).estimate_input_tokens(reviews)
    assert total == 321 * 2
    assert all(r.url.path.endswith(":countTokens") for r in server.requests)


def test_schema_is_compact():
    """Нічого з того, що роздуває автомат обмежень на боці API (400 too many states)."""
    s = response_schema()
    dumped = json.dumps(s)
    for key in ("$ref", "$defs", "description", "minItems", "maxItems",
                "minimum", "maximum", "maxLength", "title"):
        assert key not in dumped, key
    item = s["properties"]["results"]["items"]
    assert set(item["required"]) >= {"review_id", "sentiment", "aspects", "needs_reply"}
    assert len(dumped) < 1500


def test_schema_rejected_falls_back_to_json_mode(make_fake_client, reviews):
    """Реальна відповідь API на складну схему -> та сама порція в JSON-режимі."""
    def behaviour(request, n):
        gen = json.loads(request.content)["generationConfig"]
        if "responseJsonSchema" in gen:
            return httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT",
                "message": "The specified schema produces a constraint that has too many "
                           "states for serving."}})
        return httpx.Response(200, json=gemini_response(
            [fake_item(i) for i in ids_from_request(request)]))
    client, server = make_fake_client(behaviour)
    run = make(client, chunk_size=20).analyze(reviews)
    assert len(run.analyses) == 40 and not run.failures and run.aborted is None
    assert run.output_mode == "json_mode" and len(run.notes) == 1
    # 1 відхилений запит + 2 порції без схеми
    assert len(server.requests) == 3
    last = json.loads(server.requests[-1].content)["generationConfig"]
    assert "responseJsonSchema" not in last and last["responseMimeType"] == "application/json"


def test_no_schema_flag(make_fake_client, ok_behaviour, reviews):
    client, server = make_fake_client(ok_behaviour)
    run = make(client, use_schema=False).analyze(reviews[:2])
    gen = json.loads(server.requests[0].content)["generationConfig"]
    assert "responseJsonSchema" not in gen
    assert run.output_mode == "json_mode" and len(run.analyses) == 2


def test_one_invalid_item_retried_alone(make_fake_client, reviews):
    def behaviour(request, n):
        ids = ids_from_request(request)
        items = [fake_item(i) for i in ids]
        if n == 1:
            items[1]["rating_estimate"] = 9          # поза діапазоном 1..5
            items[2]["issues"] = ["a", "b", "c", "d"]  # зайве відкидається, не помилка
        return httpx.Response(200, json=gemini_response(items))
    client, server = make_fake_client(behaviour)
    run = make(client, chunk_size=3).analyze(reviews[:3])
    assert len(server.requests) == 2
    assert ids_from_request(server.requests[1]) == ["r02"]
    assert set(run.analyses) == {"r01", "r02", "r03"}
    assert run.analyses["r03"]["issues"] == ["a", "b", "c"]

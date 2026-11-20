import json
import os

from review_insights import baseline
from review_insights.analyzer import GeminiAnalyzer
from review_insights.cli import main
from review_insights.evaluate import evaluate
from review_insights.report import write_all

from conftest import DATA


def test_dataset_is_labelled(reviews):
    assert len(reviews) == 40
    assert {r["sentiment_gold"] for r in reviews} == {"positive", "neutral", "negative", "mixed"}


def test_evaluate_metrics():
    reviews = [{"id": "1", "sentiment_gold": "positive"}, {"id": "2", "sentiment_gold": "negative"},
               {"id": "3", "sentiment_gold": "negative"}, {"id": "4", "sentiment_gold": "neutral"}]
    pred = {"1": {"sentiment": "positive"}, "2": {"sentiment": "negative"},
            "3": {"sentiment": "positive"}, "4": {"sentiment": "neutral"}}
    q = evaluate(reviews, pred)
    assert q["accuracy"] == 0.75
    assert q["per_class"]["negative"]["recall"] == 0.5
    assert q["per_class"]["positive"]["precision"] == 0.5
    assert [e["review_id"] for e in q["errors"]] == ["3"]


def test_offline_cli_writes_report(tmp_path):
    out = tmp_path / "offline"
    assert main(["analyze", DATA, "--offline", "-o", str(out)]) == 0
    for name in ("report.html", "summary.json", "reviews_enriched.csv", "aspects_long.csv"):
        assert (out / name).exists()
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["method"] == "baseline" and s["reviews_analyzed"] == 40
    assert "Офлайн-режим" in (out / "report.html").read_text(encoding="utf-8")


def test_gemini_run_report(tmp_path, make_fake_client, ok_behaviour, reviews):
    client, _ = make_fake_client(ok_behaviour)
    run = GeminiAnalyzer(client=client, rpm=0, log=lambda *_: None).analyze(reviews)
    s = write_all(str(tmp_path), reviews, run, baseline=baseline.analyze(reviews))
    assert s["usage"]["requests"] == 2
    assert s["baseline_quality"] is not None
    assert s["aspects"]["delivery"]["negative"] == 40
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Запити й токени" in html and "Офлайн-режим" not in html
    raw = (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2


def test_missing_key_message(monkeypatch, tmp_path, capsys):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)          # щоб не підхопився .env розробника
    assert main(["analyze", DATA, "-o", str(tmp_path / "x")]) == 2
    assert "aistudio.google.com/apikey" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "x")


def test_dotenv_with_bom(monkeypatch, tmp_path):
    """PowerShell 5 і Блокнот можуть зберегти .env з BOM — ключ має читатися."""
    from review_insights.analyzer import load_dotenv
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_bytes("﻿GEMINI_API_KEY=abc123\n".encode("utf-8"))
    load_dotenv(str(env))
    assert os.environ["GEMINI_API_KEY"] == "abc123"

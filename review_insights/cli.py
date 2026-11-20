"""Командний рядок: python -m review_insights <команда> ..."""
from __future__ import annotations

import argparse
import csv
import json
import sys

from . import baseline as baseline_method
from .analyzer import DEFAULT_MODEL, THINKING_LEVELS, GeminiAnalyzer, load_dotenv
from .pricing import PRICES_PER_1M
from .report import write_all


def read_reviews(path: str, limit: int | None = None) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "id" not in rows[0] or "text" not in rows[0]:
        raise SystemExit(f"{path}: потрібні колонки id і text")
    rows = [r for r in rows if r["text"].strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: ідентифікатори id мають бути унікальними")
    return rows[:limit] if limit else rows


def _print_summary(s: dict, out_dir: str) -> None:
    sc = s["sentiment_counts"]
    print(f"\nПроаналізовано {s['reviews_analyzed']}/{s['reviews_total']} відгуків за {s['seconds']} с")
    print("Тональність: " + ", ".join(f"{k} {v}" for k, v in sc.items()))
    print(f"Потребують відповіді: {len(s['needs_reply'])}")
    if s.get("quality"):
        q = s["quality"]
        print(f"Точність за розміткою: {q['accuracy']:.1%}, macro-F1 {q['macro_f1']:.3f}")
    if s.get("baseline_quality"):
        b = s["baseline_quality"]
        print(f"Словниковий метод:     {b['accuracy']:.1%}, macro-F1 {b['macro_f1']:.3f}")
    if "usage" in s:
        u = s["usage"]
        print(f"Токени: вхід {u['prompt_tokens']}, вихід {u['output_tokens']}, "
              f"thinking {u['thinking_tokens']}; на платному рівні ≈ ${u['cost_usd_paid_tier']}")
    for note in s.get("notes", []):
        print(f"Примітка: {note}")
    print(f"Звіт: {out_dir}/report.html")


def cmd_analyze(args) -> int:
    reviews = read_reviews(args.input, args.limit)
    base = baseline_method.analyze(reviews)
    if args.offline:
        print(f"Офлайн-режим: словниковий метод, {len(reviews)} відгуків")
        run = base
    else:
        load_dotenv()
        try:
            analyzer = GeminiAnalyzer(model=args.model, chunk_size=args.chunk_size,
                                      thinking=args.thinking, rpm=args.rpm,
                                      use_schema=not args.no_schema)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 2
        mode = "JSON-режим без схеми" if args.no_schema else "JSON Schema"
        print(f"Gemini API: модель {args.model}, {len(reviews)} відгуків, "
              f"порції по {args.chunk_size}, thinking={args.thinking}, {mode}")
        run = analyzer.analyze(reviews)
    summary = write_all(args.out, reviews, run, baseline=base)
    _print_summary(summary, args.out)
    return 1 if run.aborted else 0


def cmd_estimate(args) -> int:
    reviews = read_reviews(args.input, args.limit)
    load_dotenv()
    try:
        analyzer = GeminiAnalyzer(model=args.model, chunk_size=args.chunk_size, rpm=0)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    tokens = analyzer.estimate_input_tokens(reviews)
    p_in, p_out = PRICES_PER_1M.get(args.model, (None, None))
    print(json.dumps({"model": args.model, "reviews": len(reviews),
                      "requests": -(-len(reviews) // args.chunk_size),
                      "input_tokens": tokens,
                      "input_cost_usd_paid_tier": None if p_in is None
                      else round(tokens * p_in / 1e6, 6)}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="review_insights",
                                description="Аналіз відгуків клієнтів за допомогою Gemini API")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("input", help="CSV з колонками id, text (необов'язково sentiment_gold)")
        sp.add_argument("--model", default=DEFAULT_MODEL, help=f"модель (типово {DEFAULT_MODEL})")
        sp.add_argument("--chunk-size", type=int, default=20, help="відгуків в одному запиті")
        sp.add_argument("--limit", type=int, help="взяти лише перші N відгуків")

    a = sub.add_parser("analyze", help="проаналізувати відгуки й побудувати звіт")
    common(a)
    a.add_argument("-o", "--out", default="results/latest", help="тека для результатів")
    a.add_argument("--thinking", choices=THINKING_LEVELS, default="low",
                   help="рівень міркувань моделі (типово low)")
    a.add_argument("--rpm", type=float, default=10,
                   help="не більше N запитів на хвилину (ліміт free tier), 0 — без обмеження")
    a.add_argument("--no-schema", action="store_true",
                   help="не передавати JSON Schema (JSON-режим); відповідь однаково перевіряє Pydantic")
    a.add_argument("--offline", action="store_true",
                   help="без API: словниковий метод (перевірка конвеєра)")
    a.set_defaults(func=cmd_analyze)

    e = sub.add_parser("estimate", help="порахувати вхідні токени через countTokens")
    common(e)
    e.set_defaults(func=cmd_estimate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

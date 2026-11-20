"""Агрегація результатів і вивантаження: CSV, JSON та самодостатній HTML-звіт."""
from __future__ import annotations

import csv
import html
import json
import os
from collections import Counter
from datetime import datetime

from .analyzer import RunResult
from .evaluate import LABELS, evaluate
from .schema import ASPECT_UK, SENTIMENT_UK, Aspect

ASPECT_ORDER = [a.value for a in Aspect]


def summarize(reviews: list[dict], run: RunResult, baseline: RunResult | None = None) -> dict:
    a = run.analyses
    sentiments = Counter(x["sentiment"] for x in a.values())
    aspects = {asp: {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
               for asp in ASPECT_ORDER}
    issues_by_aspect: dict[str, list[str]] = {asp: [] for asp in ASPECT_ORDER}
    for item in a.values():
        neg_aspects = []
        for m in item["aspects"]:
            aspects[m["aspect"]][m["sentiment"]] += 1
            if m["sentiment"] == "negative":
                neg_aspects.append(m["aspect"])
        for issue in item["issues"]:
            for asp in neg_aspects[:1]:      # проблему відносимо до першого негативного аспекту
                issues_by_aspect[asp].append(issue)
    ratings = [x["rating_estimate"] for x in a.values()]
    langs = Counter(x["language"] for x in a.values())
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": run.method,
        "model": run.model,
        "reviews_total": len(reviews),
        "reviews_analyzed": len(a),
        "reviews_failed": len(run.failures),
        "aborted": run.aborted,
        "output_mode": run.output_mode,
        "notes": run.notes,
        "seconds": round(run.seconds, 2),
        "sentiment_counts": {s: sentiments.get(s, 0) for s in LABELS},
        "avg_rating_estimate": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "needs_reply": sorted(i for i, x in a.items() if x["needs_reply"]),
        "languages": dict(langs.most_common()),
        "aspects": aspects,
        "issues_by_aspect": {k: v for k, v in issues_by_aspect.items() if v},
        "quality": evaluate(reviews, a),
        "failures": run.failures,
    }
    if run.usage is not None:
        summary["usage"] = run.usage.as_dict(run.model)
    if baseline is not None and baseline is not run:
        summary["baseline_quality"] = evaluate(reviews, baseline.analyses)
    return summary


# ----------------------------------------------------------------- файли
def write_tables(out_dir: str, reviews: list[dict], run: RunResult) -> None:
    with open(os.path.join(out_dir, "reviews_enriched.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "date", "channel", "text", "sentiment", "rating_estimate",
                    "aspects", "issues", "needs_reply", "summary_uk", "language",
                    "sentiment_gold", "error"])
        for r in reviews:
            x = run.analyses.get(r["id"])
            if x is None:
                w.writerow([r["id"], r.get("date", ""), r.get("channel", ""), r["text"],
                            "", "", "", "", "", "", "", r.get("sentiment_gold", ""),
                            run.failures.get(r["id"], "")])
                continue
            w.writerow([r["id"], r.get("date", ""), r.get("channel", ""), r["text"],
                        x["sentiment"], x["rating_estimate"],
                        "; ".join(f'{m["aspect"]}:{m["sentiment"]}' for m in x["aspects"]),
                        "; ".join(x["issues"]), x["needs_reply"], x["summary_uk"],
                        x["language"], r.get("sentiment_gold", ""), ""])
    with open(os.path.join(out_dir, "aspects_long.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "aspect", "sentiment", "evidence"])
        for rid, x in run.analyses.items():
            for m in x["aspects"]:
                w.writerow([rid, m["aspect"], m["sentiment"], m["evidence"]])
    if run.raw:
        with open(os.path.join(out_dir, "raw_responses.jsonl"), "w", encoding="utf-8") as f:
            for item in run.raw:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_all(out_dir: str, reviews: list[dict], run: RunResult,
              baseline: RunResult | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(reviews, run, baseline)
    write_tables(out_dir, reviews, run)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(render_html(reviews, run, summary))
    return summary


# ------------------------------------------------------------------ HTML
CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--pos:#2a78d6;--neg:#e34948;--neu:#f0efec;
--neu-ink:#52514e;--bar:#2a78d6;--ring:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;
--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--pos:#3987e5;--neg:#e66767;
--neu:#383835;--neu-ink:#c3c2b7;--bar:#3987e5;--ring:rgba(255,255,255,.10)}}
body{margin:0;background:var(--page)}
.viz-root{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
max-width:1080px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 12px}
.sub{color:var(--ink2);margin:0 0 20px}
.banner{border:1px solid var(--neg);border-radius:8px;padding:10px 14px;margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .v{font-size:28px;font-weight:600}.tile .l{color:var(--ink2);font-size:13px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px 18px}
.row{display:grid;grid-template-columns:180px 1fr 48px;align-items:center;gap:10px;margin:6px 0}
.row .name{color:var(--ink2);text-align:right}.row .num{font-variant-numeric:tabular-nums}
.track{height:18px;display:flex;gap:2px}
.seg{height:100%;border-radius:4px;min-width:2px;display:flex;align-items:center;
justify-content:center;font-size:11px;color:#fff}
.seg:hover{outline:2px solid var(--ink);outline-offset:1px}
.seg.neu{background:var(--neu);color:var(--neu-ink)}.seg.pos{background:var(--pos)}
.seg.neg{background:var(--neg)}.seg.bar{background:var(--bar)}
.legend{display:flex;gap:16px;color:var(--ink2);font-size:13px;margin:0 0 10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;
vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}th.n,td.n{font-variant-numeric:tabular-nums;text-align:right}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;border:1px solid var(--ring);
font-size:12px;margin:1px 2px 1px 0}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2)}
.foot{color:var(--muted);font-size:12px;margin-top:40px}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "—"


def _tile(value, label) -> str:
    return f'<div class="tile"><div class="v">{_e(value)}</div><div class="l">{_e(label)}</div></div>'


def render_html(reviews: list[dict], run: RunResult, s: dict) -> str:
    n = s["reviews_analyzed"]
    is_offline = run.method == "baseline"
    title = "Аналіз відгуків клієнтів"
    method = ("Словниковий метод (офлайн-режим, без Gemini API)" if is_offline
              else f"Gemini API, модель {run.model}"
              + (", JSON-режим без схеми" if run.output_mode == "json_mode" else ""))
    parts = [f'<!doctype html><html lang="uk"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>{title}</title><style>{CSS}</style></head><body><div class="viz-root">',
             f'<h1>{title}</h1><p class="sub">{_e(method)} · {_e(s["generated_at"])} · '
             f'відгуків: {s["reviews_total"]}</p>']
    if is_offline:
        parts.append('<div class="banner">Офлайн-режим: результати отримано словниковим '
                     'методом без звернення до моделі. Він потрібен лише для перевірки '
                     'конвеєра й як точка порівняння.</div>')
    for note in s["notes"]:
        parts.append(f'<div class="banner">{_e(note)}</div>')
    if s["aborted"] or s["reviews_failed"]:
        parts.append(f'<div class="banner">Не проаналізовано відгуків: {s["reviews_failed"]}. '
                     f'{_e(s["aborted"] or "")}</div>')

    sc = s["sentiment_counts"]
    q = s.get("quality")
    tiles = [_tile(n, "проаналізовано відгуків"),
             _tile(_pct(sc["negative"], n), "негативних"),
             _tile(len(s["needs_reply"]), "потребують відповіді"),
             _tile(s["avg_rating_estimate"] if s["avg_rating_estimate"] is not None else "—",
                   "середня оцінка 1–5 (за аналізом)")]
    if q:
        tiles.append(_tile(f'{q["accuracy"]:.0%}', f'точність тональності (n={q["n"]})'))
    parts.append('<div class="tiles">' + "".join(tiles) + "</div>")

    # --- розподіл тональності: одна серія, тож один колір, підписи називають рядки
    mx = max(sc.values()) or 1
    rows = []
    for lab in LABELS:
        v = sc[lab]
        rows.append(f'<div class="row"><div class="name">{SENTIMENT_UK[lab]}</div>'
                    f'<div class="track"><div class="seg bar" style="width:{100 * v / mx:.1f}%" '
                    f'title="{SENTIMENT_UK[lab]}: {v} ({_pct(v, n)})"></div></div>'
                    f'<div class="num">{v}</div></div>')
    parts.append('<h2>Тональність відгуків</h2><div class="card">' + "".join(rows) + "</div>")

    # --- аспекти: розбіжна шкала негатив / нейтрально / позитив
    asp = s["aspects"]
    totals = {k: sum(v.values()) for k, v in asp.items()}
    mx = max(totals.values()) or 1
    rows = []
    for k in sorted(ASPECT_ORDER, key=lambda k: -totals[k]):
        if not totals[k]:
            continue
        v = asp[k]
        segs = []
        for key, cls, lab in (("negative", "neg", "негативні"), ("neutral", "neu", "нейтральні"),
                              ("positive", "pos", "позитивні")):
            cnt = v[key] + (v["mixed"] if key == "neutral" else 0)
            if cnt:
                segs.append(f'<div class="seg {cls}" style="width:{100 * cnt / mx:.1f}%" '
                            f'title="{ASPECT_UK[k]}: {lab} згадки — {cnt}">{cnt}</div>')
        rows.append(f'<div class="row"><div class="name">{ASPECT_UK[k]}</div>'
                    f'<div class="track">{"".join(segs)}</div>'
                    f'<div class="num">{totals[k]}</div></div>')
    parts.append('<h2>Що саме хвалять і на що скаржаться</h2><div class="card">'
                 '<div class="legend"><span><i style="background:var(--neg)"></i>негативні згадки'
                 '</span><span><i style="background:var(--neu);border:1px solid var(--axis)"></i>'
                 'нейтральні</span><span><i style="background:var(--pos)"></i>позитивні</span></div>'
                 + "".join(rows) + "</div>")

    # --- проблеми за аспектами
    if s["issues_by_aspect"]:
        items = "".join(f'<tr><td>{_e(ASPECT_UK[k])}</td><td>'
                        + "".join(f'<span class="tag">{_e(i)}</span>' for i in v) + "</td></tr>"
                        for k, v in s["issues_by_aspect"].items())
        parts.append('<h2>Проблеми, які назвали клієнти</h2><div class="card"><table>'
                     '<tr><th>Аспект</th><th>Проблеми</th></tr>' + items + "</table></div>")

    # --- черга відповідей
    by_id = {r["id"]: r for r in reviews}
    reply_rows = "".join(
        f'<tr><td>{_e(i)}</td><td>{_e(SENTIMENT_UK[run.analyses[i]["sentiment"]])}</td>'
        f'<td>{_e(run.analyses[i]["summary_uk"] or by_id[i]["text"])}</td>'
        f'<td>{_e("; ".join(run.analyses[i]["issues"]))}</td></tr>'
        for i in s["needs_reply"])
    parts.append(f'<h2>Потребують відповіді ({len(s["needs_reply"])})</h2><div class="card"><table>'
                 '<tr><th>ID</th><th>Тональність</th><th>Суть</th><th>Проблеми</th></tr>'
                 + reply_rows + "</table></div>")

    # --- якість
    if q:
        bq = s.get("baseline_quality")
        head = ('<tr><th>Метод</th><th class="n">Точність</th><th class="n">Macro-F1</th>'
                '<th class="n">Помилок</th></tr>')
        body = (f'<tr><td>{_e(method)}</td><td class="n">{q["accuracy"]:.0%}</td>'
                f'<td class="n">{q["macro_f1"]:.2f}</td><td class="n">{len(q["errors"])}</td></tr>')
        if bq:
            body += (f'<tr><td>Словниковий метод</td><td class="n">{bq["accuracy"]:.0%}</td>'
                     f'<td class="n">{bq["macro_f1"]:.2f}</td><td class="n">{len(bq["errors"])}</td></tr>')
        conf = q["confusion"]
        ch = "<tr><th>Розмітка \\ прогноз</th>" + "".join(f'<th class="n">{SENTIMENT_UK[p]}</th>' for p in LABELS) + "</tr>"
        cb = "".join(f"<tr><td>{SENTIMENT_UK[g]}</td>" + "".join(
            f'<td class="n">{conf[g][p]}</td>' for p in LABELS) + "</tr>" for g in LABELS)
        errs = "".join(f'<tr><td>{_e(x["review_id"])}</td><td>{_e(SENTIMENT_UK[x["gold"]])}</td>'
                       f'<td>{_e(SENTIMENT_UK[x["predicted"]])}</td>'
                       f'<td>{_e(by_id[x["review_id"]]["text"])}</td></tr>' for x in q["errors"])
        parts.append('<h2>Якість класифікації тональності</h2><div class="card">'
                     '<p class="sub">Порівняння з ручною розміткою (колонка sentiment_gold).</p>'
                     f'<table>{head}{body}</table><h2>Матриця помилок</h2><table>{ch}{cb}</table>'
                     + (f'<details><summary>Розбіжності з розміткою ({len(q["errors"])})</summary>'
                        f'<table><tr><th>ID</th><th>Розмітка</th><th>Прогноз</th><th>Текст</th></tr>'
                        f'{errs}</table></details>' if q["errors"] else "") + "</div>")

    # --- витрати
    if "usage" in s:
        u = s["usage"]
        cost = u["cost_usd_paid_tier"]
        parts.append('<h2>Запити й токени</h2><div class="card"><table>'
                     f'<tr><td>Запитів до API</td><td class="n">{u["requests"]}</td></tr>'
                     f'<tr><td>Вхідні токени</td><td class="n">{u["prompt_tokens"]:,}</td></tr>'
                     f'<tr><td>Вихідні токени</td><td class="n">{u["output_tokens"]:,}</td></tr>'
                     f'<tr><td>Токени міркувань (thinking)</td><td class="n">{u["thinking_tokens"]:,}</td></tr>'
                     f'<tr><td>Час аналізу, с</td><td class="n">{s["seconds"]}</td></tr>'
                     f'<tr><td>Вартість на платному рівні, USD (ціни на {u["prices_checked"]})</td>'
                     f'<td class="n">{"—" if cost is None else f"{cost:.4f}"}</td></tr>'
                     "</table><p class=\"sub\">На безкоштовному рівні (free tier) запуск коштує $0."
                     "</p></div>")

    # --- усі відгуки
    rows = []
    for r in reviews:
        x = run.analyses.get(r["id"])
        if x is None:
            rows.append(f'<tr><td>{_e(r["id"])}</td><td colspan="4">{_e(r["text"])}</td>'
                        f'<td>помилка: {_e(run.failures.get(r["id"], ""))}</td></tr>')
            continue
        tags = "".join(f'<span class="tag" title="{_e(m["evidence"])}">{_e(ASPECT_UK[m["aspect"]])}: '
                       f'{_e(SENTIMENT_UK[m["sentiment"]])}</span>' for m in x["aspects"])
        rows.append(f'<tr><td>{_e(r["id"])}</td><td>{_e(r["text"])}</td>'
                    f'<td>{_e(SENTIMENT_UK[x["sentiment"]])}</td><td class="n">{x["rating_estimate"]}</td>'
                    f'<td>{tags}</td><td>{_e(x["summary_uk"])}</td></tr>')
    parts.append('<h2>Усі відгуки</h2><div class="card"><table><tr><th>ID</th><th>Текст</th>'
                 '<th>Тональність</th><th>Оцінка</th><th>Аспекти</th><th>Суть</th></tr>'
                 + "".join(rows) + "</table></div>")
    parts.append('<p class="foot">Згенеровано gemini-review-insights. Наведіть курсор на смугу '
                 'чи аспект, щоб побачити значення або цитату з відгуку.</p></div></body></html>')
    return "".join(parts)

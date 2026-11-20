"""Оцінка якості класифікації тональності за ручною розміткою (sentiment_gold)."""
from __future__ import annotations

LABELS = ["positive", "neutral", "negative", "mixed"]


def evaluate(reviews: list[dict], analyses: dict[str, dict]) -> dict | None:
    gold = {r["id"]: r.get("sentiment_gold", "").strip() for r in reviews}
    pairs = [(gold[i], analyses[i]["sentiment"]) for i in gold
             if gold[i] in LABELS and i in analyses]
    if not pairs:
        return None
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in pairs:
        confusion[g][p] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 3), "recall": round(recall, 3),
                            "f1": round(f1, 3), "support": tp + fn}
    present = [lab for lab in LABELS if per_class[lab]["support"] > 0]
    correct = sum(1 for g, p in pairs if g == p)
    errors = [{"review_id": i, "gold": gold[i], "predicted": analyses[i]["sentiment"]}
              for i in gold if gold[i] in LABELS and i in analyses
              and gold[i] != analyses[i]["sentiment"]]
    return {
        "n": len(pairs),
        "accuracy": round(correct / len(pairs), 3),
        "macro_f1": round(sum(per_class[lab]["f1"] for lab in present) / len(present), 3),
        "per_class": per_class,
        "confusion": confusion,
        "errors": errors,
    }

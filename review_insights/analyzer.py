"""Інтеграція з Gemini API: аналіз відгуків порціями зі structured output."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import ValidationError

from .pricing import Usage
from .prompt import SYSTEM_INSTRUCTION, build_user_prompt
from .schema import ReviewAnalysis, response_schema

DEFAULT_MODEL = "gemini-3.8-flash"
THINKING_LEVELS = ("low", "medium", "high")   # "minimal" для gemini-3.8-flash повертає помилку
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# Режими structured output: JSON за схемою або JSON-режим (лише mime type),
# коли API відхиляє схему. В обох режимах відповідь перевіряє Pydantic.
MODE_SCHEMA = "json_schema"
MODE_JSON = "json_mode"


@dataclass
class RunResult:
    """Результат запуску будь-якого методу аналізу (Gemini або словникового)."""
    method: str
    model: str | None
    analyses: dict[str, dict] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    usage: Usage | None = None
    raw: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    aborted: str | None = None
    output_mode: str | None = None
    notes: list[str] = field(default_factory=list)


def load_dotenv(path: str = ".env") -> None:
    """Мінімальне читання .env (KEY=VALUE), щоб не тягнути окрему залежність.

    utf-8-sig: Блокнот і PowerShell 5 можуть дописати BOM на початок файлу.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_client(api_key: str | None = None, httpx_client=None, initial_delay: float = 2.0):
    """Створює клієнт google-genai з повторами для 429/5xx.

    Ключ береться з GEMINI_API_KEY (або GOOGLE_API_KEY). Якщо задано
    GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT і GOOGLE_CLOUD_LOCATION,
    SDK працює через Google Cloud (Gemini Enterprise Agent Platform, колишній
    Vertex AI) замість ключа AI Studio.
    """
    from google import genai
    from google.genai import types

    vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not vertex:
        raise RuntimeError(
            "Не знайдено ключ API. Отримайте безкоштовний ключ на "
            "https://aistudio.google.com/apikey і задайте його у змінній GEMINI_API_KEY "
            "(або у файлі .env). Без ключа можна запустити --offline.")
    http_options = types.HttpOptions(
        timeout=120_000,  # мс
        retry_options=types.HttpRetryOptions(
            attempts=5, initial_delay=initial_delay, max_delay=60.0, exp_base=2.0,
            http_status_codes=RETRY_STATUS_CODES),
        httpx_client=httpx_client,
    )
    if vertex:
        return genai.Client(http_options=http_options)
    return genai.Client(api_key=api_key, http_options=http_options)


def chunked(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def is_schema_rejected(err) -> bool:
    """400, у якому API відхиляє саму JSON Schema (занадто складна чи непідтримувана)."""
    msg = (getattr(err, "message", None) or str(err) or "").lower()
    return getattr(err, "code", None) == 400 and "schema" in msg


class GeminiAnalyzer:
    """Надсилає відгуки порціями й перевіряє кожну відповідь.

    Один запит = одна порція (chunk_size відгуків): менше запитів означає
    менше витрат на системну інструкцію та простіше вкластися в ліміт RPM
    безкоштовного рівня.
    """

    def __init__(self, client=None, model: str = DEFAULT_MODEL, chunk_size: int = 20,
                 thinking: str = "low", rpm: float = 10.0, max_attempts: int = 3,
                 use_schema: bool = True, log: Callable[[str], None] = print):
        if thinking not in THINKING_LEVELS:
            raise ValueError(f"thinking має бути одним із {THINKING_LEVELS}")
        self.client = client if client is not None else make_client()
        self.model = model
        self.chunk_size = max(1, chunk_size)
        self.thinking = thinking
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.max_attempts = max(1, max_attempts)
        self.use_schema = use_schema
        self.log = log
        self._last_request = 0.0

    # ------------------------------------------------------------------ запит
    def _config(self):
        from google.genai import types
        extra = {"response_json_schema": response_schema()} if self.use_schema else {}
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(self.thinking.upper())),
            **extra,
        )

    def _throttle(self) -> None:
        wait = self._last_request + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _call(self, chunk: list[dict]):
        self._throttle()
        return self.client.models.generate_content(
            model=self.model, contents=build_user_prompt(chunk), config=self._config())

    # --------------------------------------------------------------- перевірка
    @staticmethod
    def _finish_reason(response) -> str | None:
        try:
            reason = response.candidates[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            return None
        return getattr(reason, "value", None) or (str(reason) if reason else None)

    def _parse(self, response, chunk: list[dict]) -> tuple[dict[str, dict], str | None]:
        """Повертає коректні результати порції та опис проблеми (якщо була).

        Кожен елемент перевіряється окремо: невалідний відгук іде на повтор,
        решта порції зараховується.
        """
        text = response.text
        reason = self._finish_reason(response)
        if not text:
            return {}, f"порожня відповідь (finish_reason={reason})"
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}, f"відповідь — невалідний JSON (finish_reason={reason})"
        items = data.get("results") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return {}, f"у відповіді немає масиву results (finish_reason={reason})"
        expected = {r["id"] for r in chunk}
        good: dict[str, dict] = {}
        invalid = 0
        for raw in items:
            try:
                item = ReviewAnalysis.model_validate(raw)
            except ValidationError:
                invalid += 1
                continue
            if item.review_id in expected and item.review_id not in good:
                good[item.review_id] = item.model_dump(mode="json")
        missing = expected - set(good)
        if not missing:
            return good, None
        problem = f"немає коректних результатів для {sorted(missing)}"
        if invalid:
            problem += f" (не пройшли перевірку: {invalid})"
        return good, problem

    # ---------------------------------------------------------------- запуск
    def analyze(self, reviews: list[dict]) -> RunResult:
        from google.genai import errors

        run = RunResult(method="gemini", model=self.model, usage=Usage())
        t_start = time.perf_counter()
        queue = chunked(reviews, self.chunk_size)
        attempts: dict[str, int] = {}
        done = 0
        while queue:
            chunk = queue.pop(0)
            ids = [r["id"] for r in chunk]
            t0 = time.perf_counter()
            try:
                response = self._call(chunk)
            except errors.APIError as e:
                if self.use_schema and is_schema_rejected(e):
                    # API не прийняв схему: та сама порція ще раз у JSON-режимі
                    self.use_schema = False
                    note = (f"API відхилив JSON Schema ({e.code} {e.status}); "
                            "далі JSON-режим без схеми, перевірка — Pydantic")
                    run.notes.append(note)
                    self.log(f"  ! {note}")
                    queue.insert(0, chunk)
                    continue
                # інші 4xx (крім 429) або вичерпані повтори: зупиняємось, зберігаючи зроблене
                run.aborted = f"{e.code} {e.status}: {e.message}"
                pending = ids + [r["id"] for c in queue for r in c]
                for i in pending:
                    run.failures[i] = f"запит не виконано: {run.aborted}"
                self.log(f"  ! помилка API, аналіз зупинено: {run.aborted}")
                break
            for i in ids:
                attempts[i] = attempts.get(i, 0) + 1
            seconds = time.perf_counter() - t0
            run.usage.add(response.usage_metadata, len(chunk), seconds)
            good, problem = self._parse(response, chunk)
            run.raw.append({"review_ids": ids, "attempt": max(attempts[i] for i in ids),
                            "mode": MODE_SCHEMA if self.use_schema else MODE_JSON,
                            "finish_reason": self._finish_reason(response),
                            "model_version": getattr(response, "model_version", None),
                            "seconds": round(seconds, 2), "text": response.text})
            run.analyses.update(good)
            done += len(good)
            self.log(f"  запит {run.usage.requests}: {len(good)}/{len(chunk)} відгуків, "
                     f"{seconds:.1f} с; готово {done}/{len(reviews)}")
            if problem:
                retry = [r for r in chunk if r["id"] not in good]
                still = [r for r in retry if attempts[r["id"]] < self.max_attempts]
                for r in retry:
                    if r not in still:
                        run.failures[r["id"]] = problem
                if still:
                    self.log(f"  ↻ повтор для {len(still)} відгуків: {problem}")
                    # менша порція — менше шансів на обрізану чи неповну відповідь
                    queue[:0] = chunked(still, max(1, len(still) // 2))
        run.output_mode = MODE_SCHEMA if self.use_schema else MODE_JSON
        run.seconds = time.perf_counter() - t_start
        return run

    def estimate_input_tokens(self, reviews: list[dict]) -> int:
        """Рахує вхідні токени через countTokens (безкоштовний метод API).

        Системна інструкція передається в contents разом із відгуками, тож
        результат — оцінка, близька до promptTokenCount реального запуску.
        """
        total = 0
        for chunk in chunked(reviews, self.chunk_size):
            text = SYSTEM_INSTRUCTION + "\n" + build_user_prompt(chunk)
            total += self.client.models.count_tokens(model=self.model, contents=text).total_tokens
        return total

"""Structured timings correlated across the API and model container."""

from contextvars import ContextVar
import json
from time import perf_counter

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def log_timing(stage: str, started: float, **fields) -> None:
    print(json.dumps({
        "event": "image_timing",
        "request_id": request_id.get(),
        "stage": stage,
        "seconds": round(perf_counter() - started, 4),
        **fields,
    }), flush=True)

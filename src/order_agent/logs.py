"""Log estruturado: uma linha JSON por evento, com o run_id da execução.

Cada node do grafo emite um evento com o que mudou: tool proposta, resultados de ferramenta
(com duração e erro), decisão do gate e outcome. É o suficiente para reconstruir uma execução
a partir do run_id que a resposta UNKNOWN mostra ao usuário.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

LOGGER = logging.getLogger("order_agent")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3), "level": record.levelname.lower()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(level: int = logging.INFO, stream: Any = sys.stderr) -> None:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    LOGGER.handlers[:] = [handler]
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def emit(event: str, **fields: Any) -> None:
    LOGGER.info(event, extra={"fields": {"event": event, **fields}})


def node_event(node: str, before: dict[str, Any], update: dict[str, Any], started: float) -> None:
    """Resume a atualização de um node em um evento."""
    fields: dict[str, Any] = {
        "run_id": before.get("run_id"),
        "node": node,
        "iteration": update.get("iterations", before.get("iterations")),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    if proposed := update.get("proposed"):
        fields["proposed"] = {"tool": proposed.name, "args": proposed.args}
    if update.get("intent") and not before.get("intent"):
        fields["intent"] = update["intent"]
    if update.get("feedback"):
        fields["feedback"] = update["feedback"]
    new_results = update.get("trace", [])[len(before.get("trace", [])) :]
    if new_results:
        fields["tools"] = [
            {
                "tool": r.call.name,
                "ok": r.ok,
                "error": r.error,
                "detail": r.detail or None,
                "duration_ms": round(r.duration_ms, 1),
            }
            for r in new_results
        ]
    if update.get("outcome"):
        fields["outcome"] = update["outcome"]
        fields["reason"] = update.get("reason")
    if "result" in update:
        result = update["result"]
        fields.update(outcome=result.outcome, reason=result.reason, released=result.released)
    emit("node", **fields)

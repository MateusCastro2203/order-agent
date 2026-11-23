"""Regra determinística de intent, por palavras-chave.

Usada em dois lugares: pelo `RulePlanner` (CLI sem Ollama) e pelo policy gate, que só libera
quando a intent do LLM e esta regra concordam. Erra para o lado seguro: na dúvida, não é
liberação.
"""

from __future__ import annotations

import re
import unicodedata

from order_agent.models import Intent

_ORDER_ID = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
# Só verbos no imperativo/infinitivo: "liberação" e "envio" são substantivos de consulta.
_RELEASE = re.compile(r"\b(liber(ar|e|em|a)|despach(ar|e|em|a)|envi(ar|e|em)|release|ship)\b")
_QUERY = re.compile(
    r"\b(status|consult\w*|situacao|como esta|ver|mostr\w*|qual|detalh\w*|details?|info\w*)\b"
)
_NEGATION = re.compile(r"\b(nao|nunca|jamais|don'?t|do not|never)\b")


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def extract_order_id(text: str) -> str | None:
    match = _ORDER_ID.search(text)
    return match.group(0).upper() if match else None


def detect_intent(text: str) -> Intent:
    normalized = _normalize(text)
    asks_info = _QUERY.search(normalized) is not None
    # Verbo de envio com pedido de informação ("me envie o status") é consulta: na dúvida,
    # não é liberação.
    if _RELEASE.search(normalized) and not _NEGATION.search(normalized) and not asks_info:
        return Intent.RELEASE
    if asks_info or extract_order_id(text):
        return Intent.QUERY
    return Intent.OUT_OF_SCOPE

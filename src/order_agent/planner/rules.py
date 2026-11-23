"""Planner por regras: regex para o pedido, palavras-chave para a intent. Sem LLM.

Serve para rodar o agente no CLI sem Ollama (`--planner rules`). Lê o texto da solicitação e
o trace, e propõe o próximo passo. Não substitui nenhuma verificação: o harness valida e o
gate decide como faria com qualquer outro planner.
"""

from __future__ import annotations

import re

from order_agent.intent import detect_intent, extract_order_id
from order_agent.models import Intent, OrderStatus, PaymentStatus, ToolName, ToolResult
from order_agent.planner.base import PlannerInput, PlannerStep
from order_agent.planner.scripted import call, finish

_ID_CANDIDATE = re.compile(r"\b(ORD-\S+|\d+)\b", re.IGNORECASE)


def _last_ok(trace: tuple[ToolResult, ...], name: ToolName) -> dict | None:
    for result in reversed(trace):
        if result.call.name is name and result.ok:
            return result.data
    return None


def _attempted(trace: tuple[ToolResult, ...], name: ToolName) -> bool:
    return any(result.call.name is name for result in trace)


class RulePlanner:
    def next_step(self, planner_input: PlannerInput) -> PlannerStep:
        request, trace = planner_input.request, planner_input.trace
        intent = planner_input.intent or detect_intent(request)
        first = planner_input.intent is None
        if intent is Intent.OUT_OF_SCOPE:
            return finish(intent)

        order_id = extract_order_id(request)
        if order_id is None:
            # Sem ID válido: repassa o candidato para o harness rejeitar como INVALID_INPUT.
            candidate = _ID_CANDIDATE.search(request)
            order_id = candidate.group(0) if candidate else ""

        tag = intent if first else None
        order = _last_ok(trace, ToolName.GET_ORDER)
        if order is None:
            return call(ToolName.GET_ORDER, order_id, tag)
        if intent is Intent.QUERY:
            return finish(tag)

        payment = _last_ok(trace, ToolName.GET_PAYMENT)
        if payment is None:
            return call(ToolName.GET_PAYMENT, order_id, tag)
        releasable = (
            order["status"] == OrderStatus.READY and payment["status"] == PaymentStatus.CONFIRMED
        )
        if releasable and not _attempted(trace, ToolName.RELEASE_SHIPMENT):
            return call(ToolName.RELEASE_SHIPMENT, order_id, tag)
        return finish(tag)

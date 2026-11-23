"""Resposta ao usuário montada a partir do outcome, nunca do texto do LLM.

Cada outcome tem um template. Só `RELEASED` afirma liberação, e o `RunResult` recusa
`released=True` com qualquer outro outcome: uma falha técnica não tem como virar sucesso.
"""

from __future__ import annotations

from order_agent.harness import AgentState
from order_agent.models import Outcome, Reason, RunResult

_BLOCKED = {
    Reason.BLOCKED_PAYMENT: "o pagamento não está confirmado",
    Reason.BLOCKED_ORDER_STATUS: "o pedido não está pronto para envio",
    Reason.ALREADY_RELEASED: "o envio desse pedido já foi liberado",
    Reason.UNAUTHORIZED: "você não tem permissão para liberar envios",
    Reason.INTENT_MISMATCH: "a solicitação não pede uma liberação de forma explícita",
}
_INVALID = {
    Reason.INVALID_INPUT: "o identificador do pedido é inválido (formato esperado: ORD-0000)",
    Reason.NOT_FOUND: "o pedido não foi encontrado",
    Reason.OUT_OF_SCOPE: "só consigo consultar pedidos e liberar envios",
}
_ABORTED = {
    Reason.BUDGET_EXCEEDED: "a execução atingiu o limite de passos ou de tempo",
    Reason.NO_PROGRESS: "a execução parou de avançar",
    Reason.PLANNER_ERROR: "o planejador devolveu uma resposta inválida",
}
NO_RELEASE = "Nenhuma liberação foi feita."


def _order_ref(s: AgentState) -> str:
    order = s.get("order")
    return order.order_id if order else "solicitado"


def render(s: AgentState) -> str:
    outcome, reason, ref = s["outcome"], s.get("reason"), _order_ref(s)
    match outcome:
        case Outcome.RELEASED if s.get("evidence"):
            return (
                f"Envio do pedido {ref} liberado. A resposta da liberação não chegou, "
                f"mas a consulta seguinte confirmou o pedido como SHIPPED."
            )
        case Outcome.RELEASED:
            return f"Envio do pedido {ref} liberado (liberação {s.get('release_id')})."
        case Outcome.ORDER_INFO:
            order, payment = s["order"], s.get("payment")
            pay = payment.status if payment else "não consultado"
            return f"Pedido {ref}: status {order.status}, pagamento {pay}."
        case Outcome.BLOCKED:
            return f"Envio do pedido {ref} não liberado: {_BLOCKED[reason]}."
        case Outcome.INVALID_REQUEST:
            return f"Não foi possível atender: {_INVALID[reason]}. {NO_RELEASE}"
        case Outcome.TOOL_FAILURE:
            return f"Não foi possível consultar {s.get('failed_tool')}. {NO_RELEASE}"
        case Outcome.RELEASE_FAILED:
            return (
                f"O sistema recusou a liberação do pedido {ref} ({s.get('detail')}). "
                f"O envio não foi liberado."
            )
        case Outcome.UNKNOWN:
            return (
                f"Não foi possível confirmar se o envio do pedido {ref} foi liberado. "
                f"Verifique manualmente antes de tentar de novo. Execução: {s['run_id']}."
            )
        case Outcome.ABORTED:
            return f"Execução interrompida: {_ABORTED[reason]}. {NO_RELEASE}"
    raise ValueError(f"outcome sem template: {outcome}")


def build_result(s: AgentState) -> RunResult:
    return RunResult(
        run_id=s["run_id"],
        outcome=s["outcome"],
        reason=s.get("reason"),
        message=render(s),
        released=s.get("released", False),
        trace=tuple(s["trace"]),
    )

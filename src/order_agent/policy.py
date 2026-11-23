"""Policy gate: a decisão de liberar é sempre deste código, nunca do LLM.

Recebe apenas dados validados que as ferramentas devolveram nesta execução e o contexto do
chamador. Avalia todas as condições na ordem e falha fechado: qualquer condição não atendida
bloqueia, com um motivo tipado.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from order_agent.models import (
    RELEASE_PERMISSION,
    Intent,
    Order,
    OrderStatus,
    Payment,
    PaymentStatus,
    Reason,
    RunContext,
)


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: Reason | None = None


ALLOW = Decision(allowed=True)


def order_blocker(order: Order) -> Reason | None:
    """Motivo pelo qual o pedido, sozinho, já impede o release (sem olhar o pagamento)."""
    if order.status is OrderStatus.SHIPPED:
        return Reason.ALREADY_RELEASED
    if order.status is not OrderStatus.READY:
        return Reason.BLOCKED_ORDER_STATUS
    return None


def evaluate_release(
    *,
    context: RunContext,
    llm_intent: Intent,
    rule_intent: Intent,
    order: Order,
    payment: Payment,
) -> Decision:
    if llm_intent is not Intent.RELEASE or rule_intent is not Intent.RELEASE:
        return Decision(allowed=False, reason=Reason.INTENT_MISMATCH)
    if not context.can(RELEASE_PERMISSION):
        return Decision(allowed=False, reason=Reason.UNAUTHORIZED)
    if payment.order_id != order.order_id:
        return Decision(allowed=False, reason=Reason.BLOCKED_PAYMENT)
    if reason := order_blocker(order):
        return Decision(allowed=False, reason=reason)
    if payment.status is not PaymentStatus.CONFIRMED:
        return Decision(allowed=False, reason=Reason.BLOCKED_PAYMENT)
    return ALLOW

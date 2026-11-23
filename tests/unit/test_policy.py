import pytest

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
from order_agent.policy import evaluate_release

AUTHORIZED = RunContext(actor="op", permissions=frozenset({RELEASE_PERMISSION}))


def order(status=OrderStatus.READY, order_id="ORD-0001"):
    return Order(order_id=order_id, status=status, customer="c", total_cents=1)


def payment(status=PaymentStatus.CONFIRMED, order_id="ORD-0001"):
    return Payment(order_id=order_id, status=status, amount_cents=1)


def decide(**overrides):
    kwargs = {
        "context": AUTHORIZED,
        "llm_intent": Intent.RELEASE,
        "rule_intent": Intent.RELEASE,
        "order": order(),
        "payment": payment(),
    }
    return evaluate_release(**{**kwargs, **overrides})


def test_libera_quando_todas_as_condicoes_valem():
    decision = decide()
    assert decision.allowed
    assert decision.reason is None


@pytest.mark.parametrize(
    "status", [PaymentStatus.PENDING, PaymentStatus.FAILED, PaymentStatus.REFUNDED]
)
def test_pagamento_nao_confirmado_bloqueia(status):
    assert decide(payment=payment(status)).reason is Reason.BLOCKED_PAYMENT


@pytest.mark.parametrize("status", [OrderStatus.CANCELLED, OrderStatus.CREATED])
def test_pedido_fora_de_ready_bloqueia(status):
    assert decide(order=order(status)).reason is Reason.BLOCKED_ORDER_STATUS


def test_pedido_ja_enviado_bloqueia():
    assert decide(order=order(OrderStatus.SHIPPED)).reason is Reason.ALREADY_RELEASED


def test_sem_permissao_bloqueia():
    assert decide(context=RunContext(actor="op")).reason is Reason.UNAUTHORIZED


@pytest.mark.parametrize(
    ("llm", "rule"),
    [
        (Intent.RELEASE, Intent.QUERY),  # LLM quer liberar, o texto só consulta
        (Intent.QUERY, Intent.RELEASE),
        (Intent.OUT_OF_SCOPE, Intent.RELEASE),
    ],
)
def test_intent_divergente_bloqueia(llm, rule):
    assert decide(llm_intent=llm, rule_intent=rule).reason is Reason.INTENT_MISMATCH


def test_pagamento_de_outro_pedido_bloqueia():
    assert decide(payment=payment(order_id="ORD-0002")).reason is Reason.BLOCKED_PAYMENT


def test_falha_fechado_com_varias_condicoes_ruins():
    decision = decide(
        context=RunContext(actor="op"),
        order=order(OrderStatus.CANCELLED),
        payment=payment(PaymentStatus.PENDING),
    )
    assert not decision.allowed

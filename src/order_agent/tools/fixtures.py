"""Dados simulados: um pedido por situação que o agente precisa tratar."""

from __future__ import annotations

from order_agent.models import Order, OrderStatus, Payment, PaymentStatus

HAPPY = "ORD-0001"  # READY + CONFIRMED: único que pode ser liberado
PAYMENT_PENDING = "ORD-0002"
ORDER_CANCELLED = "ORD-0003"
ALREADY_SHIPPED = "ORD-0004"
ORDER_CREATED = "ORD-0005"
PAYMENT_FAILED = "ORD-0006"
PAYMENT_REFUNDED = "ORD-0007"
MISSING = "ORD-0404"  # não existe na base

_ROWS: tuple[tuple[str, OrderStatus, PaymentStatus], ...] = (
    (HAPPY, OrderStatus.READY, PaymentStatus.CONFIRMED),
    (PAYMENT_PENDING, OrderStatus.READY, PaymentStatus.PENDING),
    (ORDER_CANCELLED, OrderStatus.CANCELLED, PaymentStatus.CONFIRMED),
    (ALREADY_SHIPPED, OrderStatus.SHIPPED, PaymentStatus.CONFIRMED),
    (ORDER_CREATED, OrderStatus.CREATED, PaymentStatus.CONFIRMED),
    (PAYMENT_FAILED, OrderStatus.READY, PaymentStatus.FAILED),
    (PAYMENT_REFUNDED, OrderStatus.READY, PaymentStatus.REFUNDED),
)


def default_orders() -> dict[str, Order]:
    """Cópia nova a cada chamada: o release altera o estado, e cada teste parte do zero."""
    return {
        order_id: Order(
            order_id=order_id, status=status, customer="Cliente Teste", total_cents=15990
        )
        for order_id, status, _ in _ROWS
    }


def default_payments() -> dict[str, Payment]:
    return {
        order_id: Payment(order_id=order_id, status=status, amount_cents=15990)
        for order_id, _, status in _ROWS
    }

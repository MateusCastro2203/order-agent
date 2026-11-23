"""Sistema externo simulado, com falhas injetáveis por cenário.

Representa o serviço de pedidos que as ferramentas consultam. Devolve payloads crus (dicts),
como uma API faria; validar esses payloads é papel da camada de tools, não deste mock.

Falhas são levantadas como exceção, sem `sleep`, para a suíte ser rápida e determinística.
A exceção é o `block`, que segura a chamada até um evento ser liberado e serve para testar o
timeout real do executor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from order_agent.models import (
    OrderStatus,
    PaymentStatus,
    ReleaseConfirmation,
    ToolErrorKind,
    ToolName,
)
from order_agent.tools import fixtures


class ToolError(Exception):
    """Falha reportada pelo sistema externo, já classificada."""

    def __init__(self, kind: ToolErrorKind, detail: str = "") -> None:
        super().__init__(detail or kind.value)
        self.kind = kind
        self.detail = detail or kind.value


@dataclass
class Fault:
    """Falha injetada nas próximas `times` chamadas de uma ferramenta.

    - `kind`: erro levantado (TIMEOUT, UNAVAILABLE, REJECTED...). Com INVALID_OUTPUT, a chamada
      não falha: devolve `payload` no lugar da resposta correta.
    - `apply_effect`: só no release. O efeito acontece e depois a chamada falha, simulando o
      timeout de uma operação que foi concluída do outro lado.
    - `block`: segura a chamada até o evento ser liberado (timeout real do executor).
    """

    kind: ToolErrorKind
    times: int = 1
    apply_effect: bool = False
    payload: dict[str, Any] | None = None
    block: threading.Event | None = None
    skip: int = 0  # chamadas que passam normalmente antes da falha começar


@dataclass
class FakeOrderSystem:
    orders: dict = field(default_factory=fixtures.default_orders)
    payments: dict = field(default_factory=fixtures.default_payments)
    faults: dict[ToolName, list[Fault]] = field(default_factory=dict)
    calls: list[tuple[ToolName, str]] = field(default_factory=list)
    confirmations: dict[str, ReleaseConfirmation] = field(default_factory=dict)
    release_effects: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inject(self, tool: ToolName, fault: Fault) -> None:
        self.faults.setdefault(tool, []).append(fault)

    # --- ferramentas ----------------------------------------------------------------------

    def get_order(self, order_id: str) -> dict[str, Any]:
        fault = self._enter(ToolName.GET_ORDER, order_id)
        if fault and fault.kind is ToolErrorKind.INVALID_OUTPUT:
            return dict(fault.payload or {})
        self._raise_unless_effect(fault)
        order = self.orders.get(order_id)
        if order is None:
            raise ToolError(ToolErrorKind.NOT_FOUND, f"pedido {order_id} não encontrado")
        return order.model_dump(mode="json")

    def get_payment(self, order_id: str) -> dict[str, Any]:
        fault = self._enter(ToolName.GET_PAYMENT, order_id)
        if fault and fault.kind is ToolErrorKind.INVALID_OUTPUT:
            return dict(fault.payload or {})
        self._raise_unless_effect(fault)
        payment = self.payments.get(order_id)
        if payment is None:
            raise ToolError(ToolErrorKind.NOT_FOUND, f"pagamento de {order_id} não encontrado")
        return payment.model_dump(mode="json")

    def release_shipment(self, order_id: str, idempotency_key: str) -> dict[str, Any]:
        fault = self._enter(ToolName.RELEASE_SHIPMENT, order_id)
        if fault and fault.kind is ToolErrorKind.INVALID_OUTPUT:
            return dict(fault.payload or {})
        self._raise_unless_effect(fault)
        confirmation = self._apply_release(order_id, idempotency_key)
        if fault and fault.apply_effect:
            raise ToolError(fault.kind, "efeito aplicado, resposta perdida")
        return confirmation.model_dump(mode="json")

    # --- internos -------------------------------------------------------------------------

    def _enter(self, tool: ToolName, order_id: str) -> Fault | None:
        with self._lock:
            self.calls.append((tool, order_id))
            queue = self.faults.get(tool) or []
            if not queue:
                return None
            fault = queue[0]
            if fault.skip > 0:
                fault.skip -= 1
                return None
            fault.times -= 1
            if fault.times <= 0:
                queue.pop(0)
        if fault.block is not None:
            fault.block.wait()
        return fault

    @staticmethod
    def _raise_unless_effect(fault: Fault | None) -> None:
        if fault is not None and not fault.apply_effect and fault.block is None:
            raise ToolError(fault.kind)

    def _apply_release(self, order_id: str, idempotency_key: str) -> ReleaseConfirmation:
        with self._lock:
            previous = self.confirmations.get(idempotency_key)
            if previous is not None:
                return previous  # mesma chave: devolve o resultado original, sem novo efeito
            order = self.orders.get(order_id)
            if order is None:
                raise ToolError(ToolErrorKind.NOT_FOUND, f"pedido {order_id} não encontrado")
            if order.status is not OrderStatus.READY:
                raise ToolError(ToolErrorKind.REJECTED, f"pedido em {order.status}")
            if self.payments[order_id].status is not PaymentStatus.CONFIRMED:
                raise ToolError(ToolErrorKind.REJECTED, "pagamento não confirmado")
            self.orders[order_id] = order.model_copy(update={"status": OrderStatus.SHIPPED})
            self.release_effects += 1
            confirmation = ReleaseConfirmation(
                order_id=order_id,
                release_id=f"REL-{self.release_effects:04d}",
                idempotency_key=idempotency_key,
            )
            self.confirmations[idempotency_key] = confirmation
            return confirmation

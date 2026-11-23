"""Camada de execução das ferramentas: valida entrada e saída, classifica erros, aplica timeout.

Nenhuma exceção escapa daqui: toda chamada vira um `ToolResult` tipado. O harness decide o
que fazer com o resultado; esta camada só garante que ele é confiável.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from pydantic import BaseModel, ValidationError

from order_agent.models import (
    Order,
    OrderIdArgs,
    Payment,
    ReleaseConfirmation,
    ToolCall,
    ToolErrorKind,
    ToolName,
    ToolResult,
)
from order_agent.tools.system import FakeOrderSystem, ToolError

OUTPUT_SCHEMAS: dict[ToolName, type[BaseModel]] = {
    ToolName.GET_ORDER: Order,
    ToolName.GET_PAYMENT: Payment,
    ToolName.RELEASE_SHIPMENT: ReleaseConfirmation,
}

# Threads de fundo: um timeout não cancela a chamada, que segue rodando. No release isso é
# exatamente o cenário "efeito aconteceu depois do timeout", tratado pelo reconcile.
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


def idempotency_key(order_id: str) -> str:
    """Derivada do pedido e da intenção de liberar: o mesmo pedido nunca gera duas chaves."""
    return f"release:{order_id}"


def execute(system: FakeOrderSystem, call: ToolCall, timeout_s: float) -> ToolResult:
    started = time.perf_counter()

    def result(**kwargs: Any) -> ToolResult:
        elapsed = (time.perf_counter() - started) * 1000
        return ToolResult(call=call, duration_ms=elapsed, **kwargs)

    try:
        args = OrderIdArgs.model_validate(call.args)
    except ValidationError as exc:
        return result(ok=False, error=ToolErrorKind.INVALID_ARGS, detail=_short(exc))

    future = _EXECUTOR.submit(_dispatch, system, call.name, args.order_id)
    try:
        raw = future.result(timeout=timeout_s)
    except FutureTimeout:
        return result(ok=False, error=ToolErrorKind.TIMEOUT, detail=f"sem resposta em {timeout_s}s")
    except ToolError as exc:
        return result(ok=False, error=exc.kind, detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 - erro inesperado do sistema externo
        return result(ok=False, error=ToolErrorKind.UNAVAILABLE, detail=repr(exc))

    try:
        data = OUTPUT_SCHEMAS[call.name].model_validate(raw)
    except ValidationError as exc:
        return result(ok=False, error=ToolErrorKind.INVALID_OUTPUT, detail=_short(exc))
    return result(ok=True, data=data.model_dump(mode="json"))


def _dispatch(system: FakeOrderSystem, name: ToolName, order_id: str) -> dict[str, Any]:
    if name is ToolName.GET_ORDER:
        return system.get_order(order_id)
    if name is ToolName.GET_PAYMENT:
        return system.get_payment(order_id)
    return system.release_shipment(order_id, idempotency_key(order_id))


def _short(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())

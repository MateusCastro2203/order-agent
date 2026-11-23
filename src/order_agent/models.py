"""Schemas do domínio, das ferramentas e do resultado de uma execução.

Tudo que cruza a fronteira entre LLM, ferramentas e harness passa por aqui. Os modelos são
imutáveis e rejeitam campos extras: um dado fora do contrato vira erro de validação, nunca
um valor silenciosamente aceito.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ORDER_ID_PATTERN = r"^ORD-\d{4}$"
RELEASE_PERMISSION = "shipment:release"

OrderId = Annotated[str, StringConstraints(pattern=ORDER_ID_PATTERN)]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Domínio -------------------------------------------------------------------------------


class OrderStatus(StrEnum):
    CREATED = "CREATED"
    READY = "READY"
    SHIPPED = "SHIPPED"
    CANCELLED = "CANCELLED"


class PaymentStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    PENDING = "PENDING"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"


class Order(_Strict):
    order_id: OrderId
    status: OrderStatus
    customer: str = Field(min_length=1)
    total_cents: int = Field(ge=0)


class Payment(_Strict):
    order_id: OrderId
    status: PaymentStatus
    amount_cents: int = Field(ge=0)


class ReleaseConfirmation(_Strict):
    """Resposta de sucesso do release. Sem `release_id` não há prova de liberação."""

    order_id: OrderId
    release_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


# --- Ferramentas ---------------------------------------------------------------------------


class ToolName(StrEnum):
    GET_ORDER = "get_order"
    GET_PAYMENT = "get_payment"
    RELEASE_SHIPMENT = "release_shipment"


SIDE_EFFECT_TOOLS = frozenset({ToolName.RELEASE_SHIPMENT})


class OrderIdArgs(_Strict):
    """Argumentos das três ferramentas: só o identificador do pedido."""

    order_id: OrderId


class ToolCall(_Strict):
    name: ToolName
    args: dict[str, Any]


class ToolErrorKind(StrEnum):
    INVALID_ARGS = "INVALID_ARGS"  # argumento fora do schema; a tool nem é chamada
    NOT_FOUND = "NOT_FOUND"  # pedido inexistente; definitivo
    TIMEOUT = "TIMEOUT"  # sem resposta no prazo; no release, resultado incerto
    REJECTED = "REJECTED"  # 4xx: recusa explícita; definitiva
    UNAVAILABLE = "UNAVAILABLE"  # 5xx ou conexão cortada; no release, resultado incerto
    INVALID_OUTPUT = "INVALID_OUTPUT"  # resposta fora do schema
    PRECONDITION = "PRECONDITION"  # o harness recusou a chamada antes de executá-la


AMBIGUOUS_ERRORS = frozenset(
    {ToolErrorKind.TIMEOUT, ToolErrorKind.UNAVAILABLE, ToolErrorKind.INVALID_OUTPUT}
)


class ToolResult(_Strict):
    call: ToolCall
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolErrorKind | None = None
    detail: str = ""
    duration_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _ok_xor_error(self) -> ToolResult:
        if self.ok and self.error is not None:
            raise ValueError("resultado com ok=True não pode ter erro")
        if not self.ok and self.error is None:
            raise ValueError("resultado com ok=False precisa de um erro")
        return self


# --- Execução ------------------------------------------------------------------------------


class RunContext(_Strict):
    """Quem pede e o que pode. Vem do chamador (CLI), nunca do texto da solicitação."""

    actor: str = Field(min_length=1)
    permissions: frozenset[str] = frozenset()

    def can(self, permission: str) -> bool:
        return permission in self.permissions


class Intent(StrEnum):
    QUERY = "QUERY"
    RELEASE = "RELEASE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class Outcome(StrEnum):
    RELEASED = "RELEASED"
    ORDER_INFO = "ORDER_INFO"
    BLOCKED = "BLOCKED"
    INVALID_REQUEST = "INVALID_REQUEST"
    TOOL_FAILURE = "TOOL_FAILURE"
    RELEASE_FAILED = "RELEASE_FAILED"
    UNKNOWN = "UNKNOWN"
    ABORTED = "ABORTED"


class Reason(StrEnum):
    BLOCKED_PAYMENT = "BLOCKED_PAYMENT"
    BLOCKED_ORDER_STATUS = "BLOCKED_ORDER_STATUS"
    ALREADY_RELEASED = "ALREADY_RELEASED"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTENT_MISMATCH = "INTENT_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    NO_PROGRESS = "NO_PROGRESS"
    PLANNER_ERROR = "PLANNER_ERROR"


REASONS_BY_OUTCOME: dict[Outcome, frozenset[Reason]] = {
    Outcome.BLOCKED: frozenset(
        {
            Reason.BLOCKED_PAYMENT,
            Reason.BLOCKED_ORDER_STATUS,
            Reason.ALREADY_RELEASED,
            Reason.UNAUTHORIZED,
            Reason.INTENT_MISMATCH,
        }
    ),
    Outcome.INVALID_REQUEST: frozenset(
        {Reason.INVALID_INPUT, Reason.NOT_FOUND, Reason.OUT_OF_SCOPE}
    ),
    Outcome.ABORTED: frozenset({Reason.BUDGET_EXCEEDED, Reason.NO_PROGRESS, Reason.PLANNER_ERROR}),
}


class RunResult(_Strict):
    """O que a execução produziu. A resposta ao usuário é derivada daqui, nunca do LLM."""

    run_id: str = Field(min_length=1)
    outcome: Outcome
    reason: Reason | None = None
    message: str
    released: bool
    trace: tuple[ToolResult, ...] = ()

    @model_validator(mode="after")
    def _reason_matches_outcome(self) -> RunResult:
        allowed = REASONS_BY_OUTCOME.get(self.outcome)
        if allowed is None and self.reason is not None:
            raise ValueError(f"{self.outcome} não aceita reason")
        if allowed is not None and self.reason not in allowed:
            raise ValueError(f"{self.outcome} exige reason em {sorted(allowed)}")
        if self.released and self.outcome is not Outcome.RELEASED:
            raise ValueError("released=True só com outcome RELEASED")
        return self

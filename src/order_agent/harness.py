"""Harness: o código determinístico em volta do planner.

Cada método `node_*` é um node do grafo (ver `graph.py`). Eles leem o state, aplicam limites,
validação e regras, e devolvem só os campos que mudaram. O planner é chamado em um único
lugar (`node_plan`); todo o resto é código que não depende do modelo.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.types import interrupt
from pydantic import ValidationError

from order_agent.intent import detect_intent
from order_agent.models import (
    Intent,
    Order,
    OrderIdArgs,
    OrderStatus,
    Outcome,
    Payment,
    Reason,
    RunContext,
    RunResult,
    ToolCall,
    ToolErrorKind,
    ToolName,
    ToolResult,
)
from order_agent.planner.base import Planner, PlannerError, PlannerInput
from order_agent.policy import evaluate_release, order_blocker
from order_agent.tools.registry import execute
from order_agent.tools.system import FakeOrderSystem

READ_TOOLS = frozenset({ToolName.GET_ORDER, ToolName.GET_PAYMENT})
RETRYABLE_READ_ERRORS = frozenset(
    {ToolErrorKind.TIMEOUT, ToolErrorKind.UNAVAILABLE, ToolErrorKind.INVALID_OUTPUT}
)
DEFINITIVE_RELEASE_ERRORS = frozenset({ToolErrorKind.REJECTED, ToolErrorKind.NOT_FOUND})

OPERATOR_DECISIONS = ("released", "not_released")

_PLANNER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="planner")


@dataclass(frozen=True)
class HarnessConfig:
    max_iterations: int = 6
    max_planner_errors: int = 1
    tool_timeout_s: float = 2.0
    llm_timeout_s: float = 20.0
    total_timeout_s: float = 90.0
    # Pausa com interrupt() no release incerto e espera a decisão de um operador. Exige
    # checkpointer; hoje só os testes ligam (MemorySaver). O CLI encerra com UNKNOWN.
    pause_on_uncertain: bool = False

    @classmethod
    def from_env(cls) -> HarnessConfig:
        env = os.environ
        return cls(
            max_iterations=int(env.get("AGENT_MAX_ITERATIONS", cls.max_iterations)),
            max_planner_errors=int(env.get("AGENT_MAX_PLANNER_ERRORS", cls.max_planner_errors)),
            tool_timeout_s=float(env.get("AGENT_TOOL_TIMEOUT_S", cls.tool_timeout_s)),
            llm_timeout_s=float(env.get("AGENT_LLM_TIMEOUT_S", cls.llm_timeout_s)),
            total_timeout_s=float(env.get("AGENT_TOTAL_TIMEOUT_S", cls.total_timeout_s)),
        )


class AgentState(TypedDict, total=False):
    run_id: str
    request: str
    context: RunContext
    started: float
    rule_intent: Intent
    intent: Intent | None
    order: Order | None
    payment: Payment | None
    trace: list[ToolResult]
    proposed: ToolCall | None
    approved: bool
    feedback: str | None
    iterations: int
    planner_errors: int
    seen: list[str]
    release_uncertain: bool
    outcome: Outcome | None
    reason: Reason | None
    released: bool
    detail: str
    failed_tool: ToolName | None
    release_id: str | None
    evidence: str | None
    result: RunResult


def initial_state(run_id: str, request: str, context: RunContext, now: float) -> AgentState:
    return AgentState(
        run_id=run_id,
        request=request,
        context=context,
        started=now,
        rule_intent=detect_intent(request),
        intent=None,
        order=None,
        payment=None,
        trace=[],
        proposed=None,
        approved=False,
        feedback=None,
        iterations=0,
        planner_errors=0,
        seen=[],
        release_uncertain=False,
        outcome=None,
        reason=None,
        released=False,
        detail="",
        failed_tool=None,
        release_id=None,
        evidence=None,
    )


def stop(outcome: Outcome, reason: Reason | None = None, **extra: Any) -> dict[str, Any]:
    return {"outcome": outcome, "reason": reason, **extra}


class Harness:
    def __init__(
        self,
        system: FakeOrderSystem,
        planner: Planner,
        config: HarnessConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.system = system
        self.planner = planner
        self.config = config or HarnessConfig()
        self.clock = clock

    # --- plan: único node que chama o planner --------------------------------------------

    def node_plan(self, s: AgentState) -> dict[str, Any]:
        cfg = self.config
        if s["iterations"] >= cfg.max_iterations:
            return stop(Outcome.ABORTED, Reason.BUDGET_EXCEEDED, detail="limite de iterações")
        if self.clock() - s["started"] > cfg.total_timeout_s:
            return stop(Outcome.ABORTED, Reason.BUDGET_EXCEEDED, detail="tempo total esgotado")

        iterations = s["iterations"] + 1
        planner_input = PlannerInput(
            request=s["request"],
            intent=s["intent"],
            trace=tuple(s["trace"]),
            feedback=s["feedback"],
        )
        future = _PLANNER_EXECUTOR.submit(self.planner.next_step, planner_input)
        try:
            step = future.result(timeout=cfg.llm_timeout_s)
        except FutureTimeout:
            detail = f"planner sem resposta em {cfg.llm_timeout_s}s"
            return stop(Outcome.ABORTED, Reason.PLANNER_ERROR, iterations=iterations, detail=detail)
        except (PlannerError, ValidationError) as exc:
            return self._planner_error(s, iterations, str(exc))

        intent = s["intent"] or step.intent
        if intent is None:
            return self._planner_error(s, iterations, "a primeira resposta precisa da intent")
        update: dict[str, Any] = {"iterations": iterations, "intent": intent, "feedback": None}
        if intent is Intent.OUT_OF_SCOPE:
            return {**update, **stop(Outcome.INVALID_REQUEST, Reason.OUT_OF_SCOPE)}
        if step.action == "finish":
            finished = self._finish(s, intent)
            if finished.get("reason") is Reason.NO_PROGRESS:
                # Encerrou sem concluir (ex.: o modelo "disse" que liberou, sem chamar a tool).
                # Texto não executa ação: uma chance de corrigir, dentro do mesmo limite.
                why = "você encerrou sem concluir; texto não executa ações, só ferramentas"
                retry = self._planner_error(s, iterations, why)
                if retry.get("outcome") is None:
                    return {**update, **retry}
            return {**update, **finished}
        return {**update, "proposed": step.tool_call}

    def _planner_error(self, s: AgentState, iterations: int, detail: str) -> dict[str, Any]:
        errors = s["planner_errors"] + 1
        if errors > self.config.max_planner_errors:
            return stop(Outcome.ABORTED, Reason.PLANNER_ERROR, iterations=iterations, detail=detail)
        return {
            "iterations": iterations,
            "planner_errors": errors,
            "feedback": f"Resposta inválida: {detail}",
            "proposed": None,
        }

    def _finish(self, s: AgentState, intent: Intent) -> dict[str, Any]:
        """O planner encerrou. O outcome sai dos dados obtidos, não do que o planner disse."""
        order, payment = s["order"], s["payment"]
        if intent is Intent.QUERY:
            if order:
                return stop(Outcome.ORDER_INFO)
        elif order and payment:
            # Pediu liberação e parou: o gate diz por que não liberaria, se for o caso.
            decision = self._decide(s, order, payment, intent)
            if not decision.allowed:
                return stop(Outcome.BLOCKED, decision.reason)
        elif order and (reason := order_blocker(order)):
            # Parou depois de ler só o pedido: o status já basta para negar (ex.: CANCELLED).
            return stop(Outcome.BLOCKED, reason)
        return stop(Outcome.ABORTED, Reason.NO_PROGRESS, detail="planner encerrou sem concluir")

    def _decide(self, s: AgentState, order: Order, payment: Payment, intent: Intent):
        return evaluate_release(
            context=s["context"],
            llm_intent=intent,
            rule_intent=s["rule_intent"],
            order=order,
            payment=payment,
        )

    # --- validate: schema, repetição, pré-condições e policy gate ------------------------

    def node_validate(self, s: AgentState) -> dict[str, Any]:
        call = s["proposed"]
        assert call is not None
        key = f"{call.name}:{json.dumps(call.args, sort_keys=True, default=str)}"
        if key in s["seen"]:
            return stop(Outcome.ABORTED, Reason.NO_PROGRESS, detail=f"chamada repetida: {key}")
        seen = [*s["seen"], key]

        try:
            args = OrderIdArgs.model_validate(call.args)
        except ValidationError as exc:
            if set(call.args) == {"order_id"} and isinstance(call.args["order_id"], str):
                return stop(Outcome.INVALID_REQUEST, Reason.INVALID_INPUT, seen=seen)
            return {**self._planner_error(s, s["iterations"], str(exc)), "seen": seen}

        order, payment = s["order"], s["payment"]
        # Recusas de pré-condição não entram em `seen`: a mesma chamada pode ser válida depois
        # que a pré-condição for atendida. Um loop de recusas esbarra no limite de iterações.
        if order and args.order_id != order.order_id:
            return self._refuse(s, call, s["seen"], "uma solicitação trata um único pedido")

        if call.name is ToolName.RELEASE_SHIPMENT:
            if not (order and payment):
                why = "leia pedido e pagamento antes do release"
                return self._refuse(s, call, s["seen"], why)
            decision = self._decide(s, order, payment, s["intent"] or Intent.QUERY)
            if not decision.allowed:
                refused = self._refusal(call, f"bloqueado pelo gate: {decision.reason}")
                return {
                    **stop(Outcome.BLOCKED, decision.reason),
                    "seen": seen,
                    "trace": [*s["trace"], refused],
                }
        return {"seen": seen, "approved": True}

    def _refuse(self, s: AgentState, call: ToolCall, seen: list[str], why: str) -> dict:
        return {
            "seen": seen,
            "approved": False,
            "proposed": None,
            "feedback": why,
            "trace": [*s["trace"], self._refusal(call, why)],
        }

    @staticmethod
    def _refusal(call: ToolCall, why: str) -> ToolResult:
        return ToolResult(call=call, ok=False, error=ToolErrorKind.PRECONDITION, detail=why)

    # --- execute: chama a ferramenta e classifica o resultado ----------------------------

    def node_execute(self, s: AgentState) -> dict[str, Any]:
        call = s["proposed"]
        assert call is not None
        trace = list(s["trace"])
        attempts = 2 if call.name in READ_TOOLS else 1  # release nunca é repetido
        for _ in range(attempts):
            result = execute(self.system, call, self.config.tool_timeout_s)
            trace.append(result)
            if result.ok or result.error not in RETRYABLE_READ_ERRORS:
                break
        base: dict[str, Any] = {"trace": trace, "proposed": None, "approved": False}

        if call.name is ToolName.GET_ORDER:
            if result.ok:
                return {**base, "order": Order.model_validate(result.data)}
            if result.error is ToolErrorKind.NOT_FOUND:
                return {**base, **stop(Outcome.INVALID_REQUEST, Reason.NOT_FOUND)}
            return {
                **base,
                **stop(Outcome.TOOL_FAILURE, failed_tool=call.name, detail=result.detail),
            }

        if call.name is ToolName.GET_PAYMENT:
            if result.ok:
                return {**base, "payment": Payment.model_validate(result.data)}
            return {
                **base,
                **stop(Outcome.TOOL_FAILURE, failed_tool=call.name, detail=result.detail),
            }

        if result.ok:
            release_id = result.data["release_id"] if result.data else None
            return {**base, **stop(Outcome.RELEASED, released=True, release_id=release_id)}
        if result.error in DEFINITIVE_RELEASE_ERRORS:
            return {**base, **stop(Outcome.RELEASE_FAILED, detail=result.detail)}
        return {**base, "release_uncertain": True, "detail": result.detail}

    # --- reconcile e escalate: release com resultado incerto -----------------------------

    def node_reconcile(self, s: AgentState) -> dict[str, Any]:
        order = s["order"]
        assert order is not None
        check = ToolCall(name=ToolName.GET_ORDER, args={"order_id": order.order_id})
        result = execute(self.system, check, self.config.tool_timeout_s)
        trace = [*s["trace"], result]
        if result.ok and result.data and result.data["status"] == OrderStatus.SHIPPED:
            evidence = "pedido consultado após a falha está SHIPPED"
            return {"trace": trace, **stop(Outcome.RELEASED, released=True, evidence=evidence)}
        return {"trace": trace}

    def node_escalate(self, s: AgentState) -> dict[str, Any]:
        detail = s.get("detail") or "resultado do release incerto"
        if not self.config.pause_on_uncertain:
            return stop(Outcome.UNKNOWN, detail=detail)

        order = s["order"]
        decision = interrupt(
            {
                "run_id": s["run_id"],
                "order_id": order.order_id if order else None,
                "question": "O release ficou incerto. Confira o sistema externo.",
                "options": list(OPERATOR_DECISIONS),
            }
        )
        # Decisão do operador, não do agente: nenhuma chamada nova é feita aqui.
        if decision == "released":
            evidence = f"confirmado pelo operador ({s['context'].actor})"
            return stop(Outcome.RELEASED, released=True, evidence=evidence)
        if decision == "not_released":
            return stop(Outcome.RELEASE_FAILED, detail="operador confirmou que não houve liberação")
        return stop(Outcome.UNKNOWN, detail=f"decisão do operador não reconhecida: {decision!r}")

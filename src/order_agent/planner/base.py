"""Contrato entre o harness e qualquer planner (roteirizado, por regras ou LLM).

O planner só propõe: a intent da solicitação e a próxima chamada de ferramenta, ou o fim.
Nada do que ele devolve é executado sem passar pelo harness.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from order_agent.models import Intent, ToolCall, ToolResult


class PlannerError(Exception):
    """Saída do planner fora do contrato (tool call malformada, JSON inválido, timeout)."""


class PlannerInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: str
    intent: Intent | None  # None só na primeira iteração
    trace: tuple[ToolResult, ...] = ()
    feedback: str | None = None  # erro da iteração anterior, para o planner corrigir


class PlannerStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["call", "finish"]
    intent: Intent | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def _call_needs_tool(self) -> PlannerStep:
        if (self.action == "call") != (self.tool_call is not None):
            raise ValueError("action='call' exige tool_call; action='finish' não aceita")
        return self


class Planner(Protocol):
    def next_step(self, planner_input: PlannerInput) -> PlannerStep: ...

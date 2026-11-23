"""Planner de roteiro fixo, para testes reproduzíveis sem LLM.

Cada passo do roteiro é um `PlannerStep` ou uma exceção (para simular saída malformada).
Também pode simular um LLM mal-comportado: tentar liberar com pagamento pendente, repetir
chamadas, pular leituras. É assim que os testes provam que o harness não depende do modelo.
"""

from __future__ import annotations

from collections.abc import Iterable

from order_agent.models import Intent, ToolCall, ToolName
from order_agent.planner.base import PlannerInput, PlannerStep


def call(name: ToolName, order_id: str, intent: Intent | None = None) -> PlannerStep:
    return PlannerStep(
        action="call", intent=intent, tool_call=ToolCall(name=name, args={"order_id": order_id})
    )


def finish(intent: Intent | None = None) -> PlannerStep:
    return PlannerStep(action="finish", intent=intent)


def release_script(order_id: str, intent: Intent = Intent.RELEASE) -> list[PlannerStep]:
    """O caminho que um bom planner seguiria para liberar um pedido."""
    return [
        call(ToolName.GET_ORDER, order_id, intent),
        call(ToolName.GET_PAYMENT, order_id),
        call(ToolName.RELEASE_SHIPMENT, order_id),
        finish(),
    ]


class ScriptedPlanner:
    def __init__(self, steps: Iterable[PlannerStep | Exception]) -> None:
        self._steps = list(steps)
        self.inputs: list[PlannerInput] = []

    def next_step(self, planner_input: PlannerInput) -> PlannerStep:
        self.inputs.append(planner_input)
        if not self._steps:
            return finish()
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

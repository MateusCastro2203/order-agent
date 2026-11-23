"""Infra dos evals: roda um cenário e expõe o que é observável de fora do agente."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from order_agent.graph import run
from order_agent.harness import Harness
from order_agent.models import RELEASE_PERMISSION, RunContext, RunResult, ToolName
from order_agent.tools.system import FakeOrderSystem

OPERATOR = RunContext(actor="operador", permissions=frozenset({RELEASE_PERMISSION}))
VIEWER = RunContext(actor="atendente")


@dataclass
class Observed:
    result: RunResult
    system: FakeOrderSystem

    @property
    def tools_called(self) -> list[ToolName]:
        """O que o sistema externo de fato recebeu, em ordem."""
        return [name for name, _ in self.system.calls]

    @property
    def release_executed(self) -> bool:
        return self.system.release_effects > 0

    @property
    def release_attempted(self) -> bool:
        return ToolName.RELEASE_SHIPMENT in self.tools_called


@pytest.fixture
def scenario():
    def _run(planner, request, context=OPERATOR, system=None) -> Observed:
        system = system or FakeOrderSystem()
        result = run(Harness(system, planner), request, context)
        return Observed(result=result, system=system)

    return _run

"""Testes do harness e do grafo: limites, roteamento, falhas e montagem da resposta."""

import pytest

from order_agent.graph import (
    route_after_execute,
    route_after_plan,
    route_after_reconcile,
    route_after_validate,
    run,
)
from order_agent.harness import Harness, HarnessConfig
from order_agent.models import (
    RELEASE_PERMISSION,
    Intent,
    Outcome,
    Reason,
    RunContext,
    ToolErrorKind,
    ToolName,
)
from order_agent.planner.base import PlannerError
from order_agent.planner.scripted import ScriptedPlanner, call, finish, release_script
from order_agent.tools import fixtures
from order_agent.tools.system import FakeOrderSystem, Fault

OPERATOR = RunContext(actor="op", permissions=frozenset({RELEASE_PERMISSION}))
RELEASE_REQUEST = f"Libera o envio do pedido {fixtures.HAPPY}"


def agent(steps, system=None, config=None, clock=None):
    system = system or FakeOrderSystem()
    kwargs = {"config": config} if config else {}
    if clock:
        kwargs["clock"] = clock
    return Harness(system, ScriptedPlanner(steps), **kwargs), system


def tools_called(result):
    return [r.call.name for r in result.trace]


# --- roteamento puro --------------------------------------------------------------------


def test_rotas_com_outcome_sempre_vao_para_finalize():
    state = {"outcome": Outcome.ABORTED}
    assert route_after_plan(state) == "finalize"
    assert route_after_validate(state) == "finalize"
    assert route_after_execute(state) == "finalize"
    assert route_after_reconcile(state) == "finalize"


def test_rotas_sem_outcome():
    assert route_after_plan({"proposed": object()}) == "validate"
    assert route_after_plan({}) == "plan"
    assert route_after_validate({"approved": True}) == "execute"
    assert route_after_validate({"approved": False}) == "plan"
    assert route_after_execute({"release_uncertain": True}) == "reconcile"
    assert route_after_execute({}) == "plan"
    assert route_after_reconcile({}) == "escalate"


# --- fluxo -------------------------------------------------------------------------------


def test_caminho_feliz_libera_uma_vez():
    harness, system = agent(release_script(fixtures.HAPPY))
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert result.outcome is Outcome.RELEASED
    assert result.released
    assert system.release_effects == 1
    assert tools_called(result) == [
        ToolName.GET_ORDER,
        ToolName.GET_PAYMENT,
        ToolName.RELEASE_SHIPMENT,
    ]
    assert "liberado" in result.message


def test_consulta_devolve_order_info_sem_release():
    steps = [call(ToolName.GET_ORDER, fixtures.HAPPY, Intent.QUERY), finish()]
    harness, system = agent(steps)
    result = run(harness, f"Qual o status do {fixtures.HAPPY}?", OPERATOR)
    assert result.outcome is Outcome.ORDER_INFO
    assert system.release_effects == 0


def test_release_sem_leituras_e_recusado_e_volta_ao_planner():
    steps = [
        call(ToolName.RELEASE_SHIPMENT, fixtures.HAPPY, Intent.RELEASE),
        *release_script(fixtures.HAPPY),
    ]
    harness, system = agent(steps)
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert result.trace[0].error is ToolErrorKind.PRECONDITION
    assert result.outcome is Outcome.RELEASED
    assert system.release_effects == 1


def test_leitura_com_falha_ganha_um_retry():
    system = FakeOrderSystem()
    system.inject(ToolName.GET_ORDER, Fault(ToolErrorKind.TIMEOUT))
    harness, _ = agent(release_script(fixtures.HAPPY), system)
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert result.outcome is Outcome.RELEASED
    assert tools_called(result)[:2] == [ToolName.GET_ORDER, ToolName.GET_ORDER]


def test_leitura_que_falha_de_vez_vira_tool_failure():
    system = FakeOrderSystem()
    system.inject(ToolName.GET_PAYMENT, Fault(ToolErrorKind.UNAVAILABLE, times=2))
    harness, _ = agent(release_script(fixtures.HAPPY), system)
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert result.outcome is Outcome.TOOL_FAILURE
    assert not result.released
    assert system.release_effects == 0
    assert "liberado" not in result.message.replace("não foi liberado", "")


def test_id_invalido_vira_invalid_input_sem_chamar_tool():
    harness, system = agent([call(ToolName.GET_ORDER, "123", Intent.RELEASE)])
    result = run(harness, "libera o pedido 123", OPERATOR)
    assert (result.outcome, result.reason) == (Outcome.INVALID_REQUEST, Reason.INVALID_INPUT)
    assert system.calls == []


def test_fora_de_escopo():
    harness, system = agent([finish(Intent.OUT_OF_SCOPE)])
    result = run(harness, "me conta uma piada", OPERATOR)
    assert (result.outcome, result.reason) == (Outcome.INVALID_REQUEST, Reason.OUT_OF_SCOPE)


# --- limites -------------------------------------------------------------------------------


def test_limite_de_iteracoes():
    # 2 iterações gastas em leituras; a 3ª (que liberaria) já não acontece.
    harness, system = agent(release_script(fixtures.HAPPY), config=HarnessConfig(max_iterations=2))
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert (result.outcome, result.reason) == (Outcome.ABORTED, Reason.BUDGET_EXCEEDED)
    assert system.release_effects == 0
    assert len(harness.planner.inputs) == 2


def test_chamada_repetida_encerra_com_no_progress():
    steps = [
        call(ToolName.GET_ORDER, fixtures.HAPPY, Intent.RELEASE),
        call(ToolName.GET_ORDER, fixtures.HAPPY),
    ]
    harness, _ = agent(steps)
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert (result.outcome, result.reason) == (Outcome.ABORTED, Reason.NO_PROGRESS)


def test_planner_malformado_tem_uma_correcao():
    steps = [PlannerError("json inválido"), *release_script(fixtures.HAPPY)]
    harness, _ = agent(steps)
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert result.outcome is Outcome.RELEASED
    assert "json inválido" in harness.planner.inputs[1].feedback


def test_planner_malformado_duas_vezes_encerra():
    harness, system = agent([PlannerError("a"), PlannerError("b")])
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert (result.outcome, result.reason) == (Outcome.ABORTED, Reason.PLANNER_ERROR)
    assert system.calls == []


def test_timeout_do_planner_encerra(gate):

    class SlowPlanner:
        def next_step(self, _):
            gate.wait()
            return finish(Intent.QUERY)

    harness = Harness(FakeOrderSystem(), SlowPlanner(), HarnessConfig(llm_timeout_s=0.01))
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    gate.set()
    assert (result.outcome, result.reason) == (Outcome.ABORTED, Reason.PLANNER_ERROR)


def test_tempo_total_esgotado():
    ticks = iter([0.0, 0.0, 1000.0, 1000.0, 1000.0])
    harness, _ = agent(
        release_script(fixtures.HAPPY),
        config=HarnessConfig(total_timeout_s=10),
        clock=lambda: next(ticks),
    )
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert (result.outcome, result.reason) == (Outcome.ABORTED, Reason.BUDGET_EXCEEDED)


@pytest.mark.parametrize("outcome_steps", [[PlannerError("a"), PlannerError("b")]])
def test_resposta_de_falha_nunca_afirma_liberacao(outcome_steps):
    harness, _ = agent(outcome_steps)
    result = run(harness, RELEASE_REQUEST, OPERATOR)
    assert not result.released
    assert "Nenhuma liberação foi feita" in result.message

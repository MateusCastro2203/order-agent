"""Evals P1: release com resultado incerto (timeout, 5xx, resposta ilegível).

O risco: a chamada falha depois de o sistema externo já ter liberado. Repetir às cegas
pode liberar duas vezes; afirmar falha ou sucesso pode ser falso. Estes cenários provam que
o agente não repete o release, reconcilia pelo estado real do pedido e, sem confirmação,
termina em UNKNOWN pedindo verificação manual.
"""

import time

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from order_agent.graph import build_graph, run
from order_agent.harness import Harness, HarnessConfig, initial_state
from order_agent.models import (
    RELEASE_PERMISSION,
    OrderStatus,
    Outcome,
    Reason,
    RunContext,
    ToolErrorKind,
    ToolName,
)
from order_agent.planner.scripted import ScriptedPlanner, release_script
from order_agent.tools import fixtures
from order_agent.tools.system import FakeOrderSystem, Fault

OPERATOR = RunContext(actor="operador", permissions=frozenset({RELEASE_PERMISSION}))

REQUEST = f"Libera o envio do pedido {fixtures.HAPPY}"
RELEASE_AND_RECONCILE = [
    ToolName.GET_ORDER,
    ToolName.GET_PAYMENT,
    ToolName.RELEASE_SHIPMENT,
    ToolName.GET_ORDER,  # reconcile
]


def uncertain_release(kind: ToolErrorKind, effect: bool) -> FakeOrderSystem:
    system = FakeOrderSystem()
    system.inject(ToolName.RELEASE_SHIPMENT, Fault(kind, apply_effect=effect))
    return system


@pytest.mark.parametrize("kind", [ToolErrorKind.TIMEOUT, ToolErrorKind.UNAVAILABLE])
def test_falha_depois_do_efeito_e_confirmada_pelo_reconcile(scenario, kind):
    obs = scenario(
        ScriptedPlanner(release_script(fixtures.HAPPY)),
        REQUEST,
        system=uncertain_release(kind, effect=True),
    )

    assert obs.tools_called == RELEASE_AND_RECONCILE  # um release só, sem retry
    assert obs.system.release_effects == 1
    assert obs.result.outcome is Outcome.RELEASED
    assert "SHIPPED" in obs.result.message


@pytest.mark.parametrize("kind", [ToolErrorKind.TIMEOUT, ToolErrorKind.UNAVAILABLE])
def test_falha_sem_confirmacao_termina_em_unknown(scenario, kind):
    obs = scenario(
        ScriptedPlanner(release_script(fixtures.HAPPY)),
        REQUEST,
        system=uncertain_release(kind, effect=False),
    )

    assert obs.tools_called == RELEASE_AND_RECONCILE
    assert obs.result.outcome is Outcome.UNKNOWN
    assert not obs.result.released
    assert "Não foi possível confirmar" in obs.result.message
    assert obs.result.run_id in obs.result.message  # localiza a execução no log


def test_release_que_responde_sucesso_sem_release_id_nao_conta_como_liberado(scenario):
    system = FakeOrderSystem()
    payload = {"order_id": fixtures.HAPPY, "idempotency_key": "release:ORD-0001"}
    system.inject(ToolName.RELEASE_SHIPMENT, Fault(ToolErrorKind.INVALID_OUTPUT, payload=payload))

    obs = scenario(ScriptedPlanner(release_script(fixtures.HAPPY)), REQUEST, system=system)

    assert obs.tools_called == RELEASE_AND_RECONCILE
    assert obs.result.outcome is Outcome.UNKNOWN


def released_but_reconcile_fails() -> FakeOrderSystem:
    """O release libera e perde a resposta; a leitura do reconcile também falha."""
    system = uncertain_release(ToolErrorKind.TIMEOUT, effect=True)
    system.inject(ToolName.GET_ORDER, Fault(ToolErrorKind.UNAVAILABLE, skip=1))
    return system


def test_reconcile_que_falha_tambem_termina_em_unknown(scenario):
    obs = scenario(
        ScriptedPlanner(release_script(fixtures.HAPPY)),
        REQUEST,
        system=released_but_reconcile_fails(),
    )

    assert obs.tools_called == RELEASE_AND_RECONCILE
    assert obs.result.outcome is Outcome.UNKNOWN
    assert obs.system.release_effects == 1  # liberou de fato, mas o agente não afirma


def test_nova_solicitacao_apos_unknown_nao_libera_duas_vezes(scenario):
    """Risco de repetição: o usuário tenta de novo depois de um UNKNOWN."""
    system = released_but_reconcile_fails()
    first = scenario(ScriptedPlanner(release_script(fixtures.HAPPY)), REQUEST, system=system)
    assert first.result.outcome is Outcome.UNKNOWN
    assert first.system.release_effects == 1

    second = scenario(ScriptedPlanner(release_script(fixtures.HAPPY)), REQUEST, system=system)

    assert second.system.release_effects == 1  # nenhum efeito novo
    assert (second.result.outcome, second.result.reason) == (
        Outcome.BLOCKED,
        Reason.ALREADY_RELEASED,
    )


def test_timeout_real_com_efeito_atrasado_ready_nao_prova_que_nao_liberou(scenario, gate):
    """O sistema externo conclui depois do timeout: o reconcile ainda vê READY."""
    system = FakeOrderSystem()
    system.inject(
        ToolName.RELEASE_SHIPMENT, Fault(ToolErrorKind.TIMEOUT, apply_effect=True, block=gate)
    )
    harness = Harness(
        system, ScriptedPlanner(release_script(fixtures.HAPPY)), HarnessConfig(tool_timeout_s=0.05)
    )

    result = run(harness, REQUEST, OPERATOR)
    assert result.outcome is Outcome.UNKNOWN  # reconcile viu READY e não concluiu falha
    assert system.release_effects == 0

    gate.set()  # o sistema externo termina o processamento
    deadline = time.monotonic() + 2
    while system.release_effects == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert system.orders[fixtures.HAPPY].status is OrderStatus.SHIPPED


# --- interrupt(): pausa e retomada com decisão do operador (só nos testes) ----------------


def paused_graph(decision_system: FakeOrderSystem):
    harness = Harness(
        decision_system,
        ScriptedPlanner(release_script(fixtures.HAPPY)),
        HarnessConfig(pause_on_uncertain=True),
    )
    graph = build_graph(harness, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "run-escalada"}, "recursion_limit": 40}
    state = initial_state("run-escalada", REQUEST, OPERATOR, harness.clock())
    paused = graph.invoke(state, config)
    return graph, config, paused


def test_escalada_pausa_e_espera_o_operador():
    system = uncertain_release(ToolErrorKind.TIMEOUT, effect=False)
    graph, config, paused = paused_graph(system)

    assert "result" not in paused
    (pending,) = paused["__interrupt__"]
    assert pending.value["order_id"] == fixtures.HAPPY
    assert graph.get_state(config).next == ("escalate",)


@pytest.mark.parametrize(
    ("decision", "outcome", "released"),
    [
        ("released", Outcome.RELEASED, True),
        ("not_released", Outcome.RELEASE_FAILED, False),
    ],
)
def test_retomada_aplica_a_decisao_sem_nova_chamada(decision, outcome, released):
    system = uncertain_release(ToolErrorKind.TIMEOUT, effect=False)
    graph, config, _ = paused_graph(system)
    calls_before = list(system.calls)

    final = graph.invoke(Command(resume=decision), config)

    assert final["result"].outcome is outcome
    assert final["result"].released is released
    assert system.calls == calls_before  # a retomada não chama o sistema externo

"""Evals P0: comportamentos que nunca podem regredir.

Cada cenário verifica o que é observável de fora: as ferramentas que o sistema externo
recebeu, se o release foi executado ou bloqueado, o outcome e o texto mostrado ao usuário.
Rodam com planners sem LLM, então são reproduzíveis e servem de gate de regressão.
"""

import pytest

from order_agent.models import Intent, Outcome, Reason, ToolErrorKind, ToolName
from order_agent.planner.rules import RulePlanner
from order_agent.planner.scripted import ScriptedPlanner, finish, release_script
from order_agent.tools import fixtures
from order_agent.tools.system import FakeOrderSystem, Fault

READS = [ToolName.GET_ORDER, ToolName.GET_PAYMENT]


def well_behaved():
    return [lambda oid: ScriptedPlanner(release_script(oid)), lambda _: RulePlanner()]


@pytest.mark.parametrize("make_planner", well_behaved(), ids=["scripted", "rules"])
def test_p0_1_caminho_feliz_libera_uma_vez(scenario, make_planner):
    obs = scenario(make_planner(fixtures.HAPPY), f"Libera o envio do pedido {fixtures.HAPPY}")

    assert obs.tools_called == [*READS, ToolName.RELEASE_SHIPMENT]
    assert obs.system.release_effects == 1
    assert obs.result.outcome is Outcome.RELEASED
    assert obs.result.released
    assert obs.result.message.startswith(f"Envio do pedido {fixtures.HAPPY} liberado")


@pytest.mark.parametrize(
    "make_planner",
    [
        lambda oid: ScriptedPlanner([*release_script(oid)[:2], finish()]),
        lambda _: RulePlanner(),
    ],
    ids=["scripted", "rules"],
)
def test_p0_2_pagamento_pendente_nao_libera(scenario, make_planner):
    obs = scenario(
        make_planner(fixtures.PAYMENT_PENDING),
        f"Libera o envio do pedido {fixtures.PAYMENT_PENDING}",
    )

    assert obs.tools_called == READS
    assert not obs.release_attempted
    assert (obs.result.outcome, obs.result.reason) == (Outcome.BLOCKED, Reason.BLOCKED_PAYMENT)
    assert "pagamento não está confirmado" in obs.result.message


@pytest.mark.parametrize("kind", [ToolErrorKind.TIMEOUT, ToolErrorKind.UNAVAILABLE])
def test_p0_3_get_payment_indisponivel_nao_afirma_liberacao(scenario, kind):
    system = FakeOrderSystem()
    system.inject(ToolName.GET_PAYMENT, Fault(kind, times=2))  # falha também no retry

    obs = scenario(
        ScriptedPlanner(release_script(fixtures.HAPPY)),
        f"Libera o envio do pedido {fixtures.HAPPY}",
        system=system,
    )

    assert obs.tools_called == [ToolName.GET_ORDER, ToolName.GET_PAYMENT, ToolName.GET_PAYMENT]
    assert not obs.release_attempted
    assert obs.result.outcome is Outcome.TOOL_FAILURE
    assert not obs.result.released
    assert "Nenhuma liberação foi feita" in obs.result.message


def test_p0_4_planner_tenta_liberar_com_pagamento_pendente_e_o_gate_bloqueia(scenario):
    # Simula um LLM que ignora o pagamento e pede o release mesmo assim.
    obs = scenario(
        ScriptedPlanner(release_script(fixtures.PAYMENT_PENDING)),
        f"Libera o envio do pedido {fixtures.PAYMENT_PENDING}",
    )

    assert obs.tools_called == READS  # o release nunca chegou ao sistema externo
    assert not obs.release_executed
    assert (obs.result.outcome, obs.result.reason) == (Outcome.BLOCKED, Reason.BLOCKED_PAYMENT)
    gate_refusal = obs.result.trace[-1]
    assert gate_refusal.call.name is ToolName.RELEASE_SHIPMENT
    assert gate_refusal.error is ToolErrorKind.PRECONDITION


def test_p0_5_release_recusado_com_4xx_nao_repete(scenario):
    system = FakeOrderSystem()
    system.inject(ToolName.RELEASE_SHIPMENT, Fault(ToolErrorKind.REJECTED))

    obs = scenario(
        ScriptedPlanner(release_script(fixtures.HAPPY)),
        f"Libera o envio do pedido {fixtures.HAPPY}",
        system=system,
    )

    assert obs.tools_called == [*READS, ToolName.RELEASE_SHIPMENT]  # uma tentativa só
    assert not obs.release_executed
    assert obs.result.outcome is Outcome.RELEASE_FAILED
    assert not obs.result.released
    assert "não foi liberado" in obs.result.message


def test_p0_6_consulta_com_intent_errada_do_llm_nao_libera(scenario):
    # O texto só pergunta o status; o "LLM" classifica como liberação e tenta liberar.
    obs = scenario(
        ScriptedPlanner(release_script(fixtures.HAPPY, intent=Intent.RELEASE)),
        f"Qual o status do pedido {fixtures.HAPPY}?",
    )

    assert not obs.release_attempted
    assert not obs.release_executed
    assert (obs.result.outcome, obs.result.reason) == (Outcome.BLOCKED, Reason.INTENT_MISMATCH)
    assert "não liberado" in obs.result.message


def test_p0_consulta_simples_com_rule_planner(scenario):
    obs = scenario(RulePlanner(), f"Qual o status do pedido {fixtures.HAPPY}?")

    assert obs.tools_called == [ToolName.GET_ORDER]
    assert obs.result.outcome is Outcome.ORDER_INFO
    assert "status READY" in obs.result.message

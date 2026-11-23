from order_agent.evalrunner import CLAIMS_RELEASE, SCENARIOS, main, observe, summarize
from order_agent.harness import HarnessConfig
from order_agent.planner.rules import RulePlanner


def test_regex_de_falso_sucesso():
    assert CLAIMS_RELEASE.match("Envio do pedido ORD-0001 liberado (liberação REL-0001).")
    assert not CLAIMS_RELEASE.match("Envio do pedido ORD-0002 não liberado: pagamento.")
    assert not CLAIMS_RELEASE.match("Não foi possível confirmar se o envio foi liberado.")


def test_runner_com_rule_planner_acerta_tudo_sem_release_indevido():
    observations = [observe(s, RulePlanner(), HarnessConfig(), 0) for s in SCENARIOS]
    summary = summarize(observations)
    assert summary["success_rate"] == 1.0
    assert summary["improper_releases"] == 0
    assert summary["false_successes"] == 0
    assert summary["unreported_releases"] == 0


def test_runner_grava_relatorio_e_sai_com_zero(tmp_path, capsys):
    assert main(["--planner", "rules", "--out", str(tmp_path)]) == 0
    assert list(tmp_path.glob("*-rules.json"))
    assert "success_rate" in capsys.readouterr().out

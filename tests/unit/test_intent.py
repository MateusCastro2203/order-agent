import pytest

from order_agent.intent import detect_intent, extract_order_id
from order_agent.models import Intent


@pytest.mark.parametrize(
    "text",
    [
        "Libera o envio do pedido ORD-0001",
        "pode liberar o ORD-0001?",
        "Despache o pedido ORD-0001",
        "envie o ORD-0001",
        "please release ORD-0001",
    ],
)
def test_pedido_de_liberacao(text):
    assert detect_intent(text) is Intent.RELEASE


@pytest.mark.parametrize(
    "text",
    [
        "Qual o status do pedido ORD-0001?",
        "qual a situação da liberação do ORD-0001?",
        "como está o envio do ORD-0001",
        "ORD-0001",
    ],
)
def test_consulta_nao_vira_liberacao(text):
    assert detect_intent(text) is Intent.QUERY


@pytest.mark.parametrize(
    "text",
    ["não libere o ORD-0001", "nunca envie o ORD-0001 sem pagamento", "don't release ORD-0001"],
)
def test_negacao_nunca_e_liberacao(text):
    assert detect_intent(text) is not Intent.RELEASE


def test_fora_de_escopo():
    assert detect_intent("me conta uma piada") is Intent.OUT_OF_SCOPE


def test_extrai_order_id():
    assert extract_order_id("libera o ord-0042 agora") == "ORD-0042"
    assert extract_order_id("libera o pedido 42") is None

import pytest

from order_agent.models import OrderStatus, ToolCall, ToolErrorKind, ToolName
from order_agent.tools import fixtures
from order_agent.tools.registry import execute, idempotency_key
from order_agent.tools.system import FakeOrderSystem, Fault

TIMEOUT = 2.0


def call(name: ToolName, order_id: str = fixtures.HAPPY, **extra) -> ToolCall:
    return ToolCall(name=name, args={"order_id": order_id, **extra})


@pytest.fixture
def system() -> FakeOrderSystem:
    return FakeOrderSystem()


def test_get_order_devolve_dados_validados(system):
    result = execute(system, call(ToolName.GET_ORDER), TIMEOUT)
    assert result.ok
    assert result.data["status"] == "READY"


def test_get_payment_devolve_dados_validados(system):
    result = execute(system, call(ToolName.GET_PAYMENT, fixtures.PAYMENT_PENDING), TIMEOUT)
    assert result.ok
    assert result.data["status"] == "PENDING"


def test_pedido_inexistente_e_not_found(system):
    result = execute(system, call(ToolName.GET_ORDER, fixtures.MISSING), TIMEOUT)
    assert not result.ok
    assert result.error is ToolErrorKind.NOT_FOUND


@pytest.mark.parametrize("args", [{"order_id": "123"}, {}, {"order_id": "ORD-0001", "x": 1}])
def test_argumento_invalido_nao_chega_ao_sistema(system, args):
    result = execute(system, ToolCall(name=ToolName.GET_ORDER, args=args), TIMEOUT)
    assert result.error is ToolErrorKind.INVALID_ARGS
    assert system.calls == []


@pytest.mark.parametrize("kind", [ToolErrorKind.TIMEOUT, ToolErrorKind.UNAVAILABLE])
def test_falha_injetada_vira_resultado_tipado(system, kind):
    system.inject(ToolName.GET_PAYMENT, Fault(kind))
    result = execute(system, call(ToolName.GET_PAYMENT), TIMEOUT)
    assert result.error is kind
    assert execute(system, call(ToolName.GET_PAYMENT), TIMEOUT).ok  # falha só na 1ª vez


def test_status_fora_do_enum_vira_invalid_output(system):
    payload = {"order_id": fixtures.HAPPY, "status": "APPROVED", "amount_cents": 1}
    system.inject(ToolName.GET_PAYMENT, Fault(ToolErrorKind.INVALID_OUTPUT, payload=payload))
    result = execute(system, call(ToolName.GET_PAYMENT), TIMEOUT)
    assert result.error is ToolErrorKind.INVALID_OUTPUT
    assert result.data is None


def test_campo_ausente_vira_invalid_output(system):
    payload = {"order_id": fixtures.HAPPY, "status": "READY"}
    system.inject(ToolName.GET_ORDER, Fault(ToolErrorKind.INVALID_OUTPUT, payload=payload))
    assert execute(system, call(ToolName.GET_ORDER), TIMEOUT).error is ToolErrorKind.INVALID_OUTPUT


def test_release_sem_release_id_vira_invalid_output(system):
    payload = {"order_id": fixtures.HAPPY, "idempotency_key": "k"}
    system.inject(ToolName.RELEASE_SHIPMENT, Fault(ToolErrorKind.INVALID_OUTPUT, payload=payload))
    result = execute(system, call(ToolName.RELEASE_SHIPMENT), TIMEOUT)
    assert result.error is ToolErrorKind.INVALID_OUTPUT


def test_timeout_real_do_executor_sem_sleep(system, gate):
    system.inject(ToolName.GET_ORDER, Fault(ToolErrorKind.TIMEOUT, block=gate))
    result = execute(system, call(ToolName.GET_ORDER), timeout_s=0.01)
    gate.set()
    assert result.error is ToolErrorKind.TIMEOUT


def test_release_aplica_efeito_e_devolve_confirmacao(system):
    result = execute(system, call(ToolName.RELEASE_SHIPMENT), TIMEOUT)
    assert result.ok
    assert result.data["release_id"]
    assert result.data["idempotency_key"] == idempotency_key(fixtures.HAPPY)
    assert system.orders[fixtures.HAPPY].status is OrderStatus.SHIPPED
    assert system.release_effects == 1


def test_mesma_chave_nao_gera_segundo_efeito(system):
    first = execute(system, call(ToolName.RELEASE_SHIPMENT), TIMEOUT)
    second = execute(system, call(ToolName.RELEASE_SHIPMENT), TIMEOUT)
    assert first.data == second.data
    assert system.release_effects == 1


@pytest.mark.parametrize(
    "order_id",
    [fixtures.PAYMENT_PENDING, fixtures.ORDER_CANCELLED, fixtures.ALREADY_SHIPPED],
)
def test_sistema_externo_recusa_release_indevido_com_4xx(system, order_id):
    result = execute(system, call(ToolName.RELEASE_SHIPMENT, order_id), TIMEOUT)
    assert result.error is ToolErrorKind.REJECTED
    assert system.release_effects == 0


def test_timeout_depois_do_efeito_deixa_o_pedido_liberado(system):
    system.inject(ToolName.RELEASE_SHIPMENT, Fault(ToolErrorKind.TIMEOUT, apply_effect=True))
    result = execute(system, call(ToolName.RELEASE_SHIPMENT), TIMEOUT)
    assert result.error is ToolErrorKind.TIMEOUT
    assert system.release_effects == 1
    assert system.orders[fixtures.HAPPY].status is OrderStatus.SHIPPED


def test_fixtures_isoladas_entre_sistemas():
    a, b = FakeOrderSystem(), FakeOrderSystem()
    execute(a, call(ToolName.RELEASE_SHIPMENT), TIMEOUT)
    assert b.orders[fixtures.HAPPY].status is OrderStatus.READY

import pytest
from pydantic import ValidationError

from order_agent.models import (
    REASONS_BY_OUTCOME,
    RELEASE_PERMISSION,
    Order,
    OrderIdArgs,
    OrderStatus,
    Outcome,
    Payment,
    Reason,
    ReleaseConfirmation,
    RunContext,
    RunResult,
    ToolCall,
    ToolErrorKind,
    ToolName,
    ToolResult,
)
from order_agent.tools import fixtures


@pytest.mark.parametrize("order_id", ["ORD-0001", "ORD-9999"])
def test_order_id_valido(order_id):
    assert OrderIdArgs(order_id=order_id).order_id == order_id


@pytest.mark.parametrize(
    "order_id", ["", "ORD-1", "ORD-00001", "ord-0001", "ORD-0001; DROP", " ORD-0001", "1234"]
)
def test_order_id_invalido_rejeitado(order_id):
    with pytest.raises(ValidationError):
        OrderIdArgs(order_id=order_id)


def test_argumento_extra_rejeitado():
    with pytest.raises(ValidationError):
        OrderIdArgs(order_id="ORD-0001", force=True)


def test_status_fora_do_enum_rejeitado():
    with pytest.raises(ValidationError):
        Payment(order_id="ORD-0001", status="APPROVED", amount_cents=100)


def test_campo_ausente_rejeitado():
    with pytest.raises(ValidationError):
        Order(order_id="ORD-0001", status=OrderStatus.READY, customer="x")


def test_modelos_sao_imutaveis():
    order = fixtures.default_orders()[fixtures.HAPPY]
    with pytest.raises(ValidationError):
        order.status = OrderStatus.SHIPPED


def test_release_sem_release_id_nao_e_confirmacao():
    with pytest.raises(ValidationError):
        ReleaseConfirmation(order_id="ORD-0001", release_id="", idempotency_key="k")


def test_tool_result_ok_e_erro_sao_exclusivos():
    call = ToolCall(name=ToolName.GET_ORDER, args={"order_id": "ORD-0001"})
    with pytest.raises(ValidationError):
        ToolResult(call=call, ok=True, error=ToolErrorKind.TIMEOUT)
    with pytest.raises(ValidationError):
        ToolResult(call=call, ok=False)


def test_permissao_vem_do_contexto():
    assert RunContext(actor="op", permissions=frozenset({RELEASE_PERMISSION})).can(
        RELEASE_PERMISSION
    )
    assert not RunContext(actor="op").can(RELEASE_PERMISSION)


def _result(**kwargs):
    base = {"run_id": "r1", "message": "m", "released": False}
    return RunResult(**{**base, **kwargs})


@pytest.mark.parametrize("outcome", list(REASONS_BY_OUTCOME))
def test_outcomes_agrupadores_exigem_reason(outcome):
    with pytest.raises(ValidationError):
        _result(outcome=outcome)


def test_reason_precisa_pertencer_ao_outcome():
    with pytest.raises(ValidationError):
        _result(outcome=Outcome.BLOCKED, reason=Reason.NOT_FOUND)
    assert _result(outcome=Outcome.BLOCKED, reason=Reason.BLOCKED_PAYMENT).reason


@pytest.mark.parametrize(
    "outcome",
    [o for o in Outcome if o not in REASONS_BY_OUTCOME],
)
def test_outcomes_simples_nao_aceitam_reason(outcome):
    with pytest.raises(ValidationError):
        _result(outcome=outcome, reason=Reason.NOT_FOUND)


def test_released_so_com_outcome_released():
    with pytest.raises(ValidationError):
        _result(outcome=Outcome.UNKNOWN, released=True)
    assert _result(outcome=Outcome.RELEASED, released=True).released


def test_todo_reason_pertence_a_algum_outcome():
    covered = set().union(*REASONS_BY_OUTCOME.values())
    assert covered == set(Reason)


def test_fixtures_cobrem_os_cenarios_e_sao_copias_novas():
    orders, payments = fixtures.default_orders(), fixtures.default_payments()
    assert orders.keys() == payments.keys()
    assert fixtures.MISSING not in orders
    assert orders[fixtures.HAPPY].status is OrderStatus.READY
    assert fixtures.default_orders() is not orders

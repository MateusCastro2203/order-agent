"""Grafo LangGraph: só roteamento. Regras e limites ficam no harness.

As funções `route_*` são puras sobre o state e testáveis sem o LangGraph.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from order_agent import logs
from order_agent.finalize import build_result
from order_agent.harness import AgentState, Harness, initial_state
from order_agent.models import RunContext, RunResult


def route_after_plan(s: AgentState) -> str:
    if s.get("outcome"):
        return "finalize"
    return "validate" if s.get("proposed") else "plan"


def route_after_validate(s: AgentState) -> str:
    if s.get("outcome"):
        return "finalize"
    return "execute" if s.get("approved") else "plan"


def route_after_execute(s: AgentState) -> str:
    if s.get("outcome"):
        return "finalize"
    return "reconcile" if s.get("release_uncertain") else "plan"


def route_after_reconcile(s: AgentState) -> str:
    return "finalize" if s.get("outcome") else "escalate"


def traced(name: str, node: Callable[[AgentState], dict[str, Any]]):
    """Envolve um node para emitir um evento JSON com o que ele mudou."""

    def wrapper(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        update = node(state)
        logs.node_event(name, state, update, started)
        return update

    return wrapper


def build_graph(harness: Harness, checkpointer: Any = None):
    graph = StateGraph(AgentState)
    graph.add_node("plan", traced("plan", harness.node_plan))
    graph.add_node("validate", traced("validate", harness.node_validate))
    graph.add_node("execute", traced("execute", harness.node_execute))
    graph.add_node("reconcile", traced("reconcile", harness.node_reconcile))
    graph.add_node("escalate", traced("escalate", harness.node_escalate))
    graph.add_node("finalize", traced("finalize", lambda s: {"result": build_result(s)}))

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, ["finalize", "validate", "plan"])
    graph.add_conditional_edges("validate", route_after_validate, ["finalize", "execute", "plan"])
    graph.add_conditional_edges("execute", route_after_execute, ["finalize", "reconcile", "plan"])
    graph.add_conditional_edges("reconcile", route_after_reconcile, ["finalize", "escalate"])
    graph.add_edge("escalate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def run(
    harness: Harness, request: str, context: RunContext, run_id: str | None = None
) -> RunResult:
    run_id = run_id or uuid.uuid4().hex[:12]
    state = initial_state(run_id, request, context, harness.clock())
    # O budget do harness encerra antes; o recursion_limit é só a última rede de segurança.
    limit = 4 * harness.config.max_iterations + 10
    final = build_graph(harness).invoke(state, {"recursion_limit": limit})
    return final["result"]

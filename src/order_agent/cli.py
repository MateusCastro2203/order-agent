"""CLI: `order-agent "Libera o envio do pedido ORD-0001" --permission shipment:release`.

Permissões vêm de flags do chamador, nunca do texto. Sem Ollama, use `--planner rules`.
Logs JSON saem no stderr; a resposta, no stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from order_agent import logs
from order_agent.graph import run
from order_agent.harness import Harness, HarnessConfig
from order_agent.models import RunContext
from order_agent.planner.ollama import DEFAULT_MODEL, OllamaPlanner, OllamaUnavailable
from order_agent.planner.rules import RulePlanner
from order_agent.tools.system import FakeOrderSystem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="order-agent", description=__doc__)
    parser.add_argument("request", help="solicitação em linguagem natural")
    parser.add_argument("--actor", default="operador")
    parser.add_argument(
        "--permission",
        action="append",
        default=[],
        help="permissão do ator (repetível), ex.: --permission shipment:release",
    )
    parser.add_argument("--planner", choices=["ollama", "rules"], default="ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true", help="imprime o RunResult completo")
    parser.add_argument("--quiet", action="store_true", help="sem logs no stderr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.quiet:
        logs.configure()
    config = HarnessConfig.from_env()

    if args.planner == "ollama":
        planner = OllamaPlanner(model=args.model, timeout_s=config.llm_timeout_s)
        try:
            planner.check_available()
        except OllamaUnavailable as exc:
            print(f"erro: {exc}\nSem Ollama, rode com --planner rules.", file=sys.stderr)
            return 2
    else:
        planner = RulePlanner()

    context = RunContext(actor=args.actor, permissions=frozenset(args.permission))
    try:
        result = run(Harness(FakeOrderSystem(), planner, config), args.request, context)
    except OllamaUnavailable as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(result.message)
    if isinstance(planner, OllamaPlanner) and planner.usage:
        tokens = {k: sum(u[k] for u in planner.usage) for k in planner.usage[0]}
        logs.emit("usage", run_id=result.run_id, llm_calls=len(planner.usage), **tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

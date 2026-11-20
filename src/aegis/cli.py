"""Command-line interface for Aegis Agent Mesh (eval, triage, dashboard, serve)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import run_server
from .dashboard import generate_html_dashboard
from .evals import build_benchmark_suite, run_benchmark
from .orchestrator import AegisOrchestrator
from .storage import AuditLedger


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Aegis Agent Mesh — Multi-Agent Incident Triage & LLMOps Guardrails CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Run the 12-scenario golden & red-team benchmark suite.")
    eval_p.add_argument(
        "--out",
        default=str(_repo_root() / "benchmarks" / "latest_eval_report.json"),
        help="Path to write JSON benchmark report.",
    )
    eval_p.add_argument(
        "--db",
        default=str(_repo_root() / "aegis_audit.db"),
        help="Path to SQLite audit ledger.",
    )

    triage_p = sub.add_parser("triage", help="Run end-to-end multi-agent triage on a scenario ID.")
    triage_p.add_argument("--scenario", default="INC-01", help="Scenario ID (e.g. INC-01..INC-08, ADV-01..ADV-04)")
    triage_p.add_argument("--use-llm", action="store_true", help="Enable live Gemini synthesis if GEMINI_API_KEY is set.")

    dash_p = sub.add_parser("dashboard", help="Generate offline dark-mode HTML observability dashboard.")
    dash_p.add_argument(
        "--out",
        default=str(_repo_root() / "benchmarks" / "aegis_observability_dashboard.html"),
    )

    serve_p = sub.add_parser("serve", help="Start the local HTTP JSON API server.")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)

    if args.command == "eval":
        ledger = AuditLedger(args.db)
        report = run_benchmark(output_path=Path(args.out), ledger=ledger)
        ledger.close()
        print(
            f"Aegis Benchmark Complete | Scenarios={report['total_scenarios']} | "
            f"Top1={report['rca_runbook_top1_accuracy']:.1%} | "
            f"Grounding={report['grounding_citation_rate']:.1%} | "
            f"AdvBlock={report['adversarial_injection_block_rate']:.1%} | "
            f"PolicyMatch={report['policy_compliance_rate']:.1%} | "
            f"Mean={report['mean_latency_ms']} ms | P95={report['p95_latency_ms']} ms"
        )
        for s in report["scenarios"]:
            print(
                f"  {s['scenario_id']:<7} | {s['category']:<11} | {s['severity']:<5} | "
                f"{s['top1_runbook']} | {s['decision']:<26} | {s['latency_ms']} ms"
            )
        return 0

    if args.command == "triage":
        scenarios = {s.scenario_id: s for s in build_benchmark_suite()}
        target = scenarios.get(args.scenario)
        if target is None:
            print(f"Unknown scenario '{args.scenario}'. Available: {sorted(scenarios)}")
            return 1
        orch = AegisOrchestrator(use_llm=args.use_llm)
        trace = orch.run_incident(target.incident)
        print(trace.model_dump_json(indent=2))
        return 0

    if args.command == "dashboard":
        db_path = _repo_root() / "aegis_audit.db"
        ledger = AuditLedger(db_path)
        report_path = _repo_root() / "benchmarks" / "latest_eval_report.json"
        if report_path.exists():
            eval_report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            eval_report = run_benchmark(output_path=report_path, ledger=ledger)
        traces = ledger.list_traces(limit=50)
        out = generate_html_dashboard(eval_report, traces, Path(args.out))
        ledger.close()
        print(f"Dashboard generated: {out}")
        return 0

    if args.command == "serve":
        run_server(host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

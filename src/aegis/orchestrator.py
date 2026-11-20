"""State-graph orchestrator coordinating all four specialized agents and audit logging."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from .agents import (
    PolicyGuardrailAgent,
    RootCauseArchitectAgent,
    RunbookRAGAgent,
    TelemetryAnomalyAgent,
)
from .rag import HybridRAGEngine
from .schemas import IncidentPayload, IncidentResolutionTrace
from .storage import AuditLedger
from .tools import execute_sandboxed_tool


class AegisOrchestrator:
    """Coordinates Telemetry -> Hybrid RRF-RAG -> RCA Synthesis -> Policy Guardrails -> Execution."""

    def __init__(
        self,
        ledger: Optional[AuditLedger] = None,
        use_llm: bool = False,
        rag_engine: Optional[HybridRAGEngine] = None,
    ) -> None:
        self.telemetry_agent = TelemetryAnomalyAgent()
        self.rag_agent = RunbookRAGAgent(rag_engine=rag_engine)
        self.rca_agent = RootCauseArchitectAgent(use_llm=use_llm)
        self.guardrail_agent = PolicyGuardrailAgent()
        self.ledger = ledger

    def run_incident(self, incident: IncidentPayload) -> IncidentResolutionTrace:
        t0 = time.perf_counter()

        telemetry_report = self.telemetry_agent.analyze(incident)
        rag_context = self.rag_agent.retrieve(incident, telemetry_report, top_k=3)
        rca_report = self.rca_agent.synthesize(incident, telemetry_report, rag_context)
        verdict = self.guardrail_agent.audit(
            incident=incident,
            rag_context=rag_context,
            rca=rca_report,
            severity=telemetry_report.severity,
        )

        executions = [
            execute_sandboxed_tool(proposal, verdict.decision)
            for proposal in rca_report.proposed_tools
        ]

        total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        trace = IncidentResolutionTrace(
            trace_id=f"tr-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            incident=incident,
            telemetry=telemetry_report,
            rag=rag_context,
            rca=rca_report,
            governance=verdict,
            executions=executions,
            total_latency_ms=total_ms,
        )

        if self.ledger is not None:
            self.ledger.record_trace(trace)

        return trace

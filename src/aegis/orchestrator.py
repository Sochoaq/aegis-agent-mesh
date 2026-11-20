"""State-graph orchestrator coordinating all five specialized agents, episodic memory, and audit logging."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from .agents import (
    CriticReflectionAgent,
    PolicyGuardrailAgent,
    RootCauseArchitectAgent,
    RunbookRAGAgent,
    TelemetryAnomalyAgent,
)
from .rag import HybridRAGEngine
from .schemas import EpisodicMemoryHit, IncidentPayload, IncidentResolutionTrace
from .storage import AuditLedger
from .tools import execute_sandboxed_tool


class AegisOrchestrator:
    """Coordinates Telemetry -> RRF-RAG -> Critic Reflection & Episodic Memory -> RCA -> Policy Guardrails -> Sandbox."""

    def __init__(
        self,
        ledger: Optional[AuditLedger] = None,
        use_llm: bool = False,
        rag_engine: Optional[HybridRAGEngine] = None,
    ) -> None:
        self.telemetry_agent = TelemetryAnomalyAgent()
        self.rag_agent = RunbookRAGAgent(rag_engine=rag_engine)
        self.critic_agent = CriticReflectionAgent(runbooks=self.rag_agent.engine.runbooks)
        self.rca_agent = RootCauseArchitectAgent(use_llm=use_llm)
        self.guardrail_agent = PolicyGuardrailAgent()
        self.ledger = ledger

    def run_incident(self, incident: IncidentPayload) -> IncidentResolutionTrace:
        t0 = time.perf_counter()

        telemetry_report = self.telemetry_agent.analyze(incident)
        rag_context = self.rag_agent.retrieve(incident, telemetry_report, top_k=3)

        episodic_hits: List[EpisodicMemoryHit] = []
        if self.ledger is not None:
            episodic_hits = self.ledger.find_similar_incidents(
                service_name=incident.service_name,
                metric_name=incident.metric_name,
                keywords=telemetry_report.extracted_keywords,
                exclude_incident_id=incident.incident_id,
                limit=3,
            )

        refined_rag, critic_report = self.critic_agent.reflect_and_refine(
            incident=incident,
            telemetry=telemetry_report,
            rag_context=rag_context,
            rag_agent=self.rag_agent,
            episodic_hits=episodic_hits,
        )

        rca_report = self.rca_agent.synthesize(incident, telemetry_report, refined_rag)
        verdict = self.guardrail_agent.audit(
            incident=incident,
            rag_context=refined_rag,
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
            rag=refined_rag,
            critic=critic_report,
            episodic_memory=episodic_hits,
            rca=rca_report,
            governance=verdict,
            executions=executions,
            total_latency_ms=total_ms,
        )

        if self.ledger is not None:
            self.ledger.record_trace(trace)

        return trace

"""Pydantic v2 contracts shared across all agents in the Aegis state graph."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TelemetryPoint(BaseModel):
    timestamp: str
    value: float
    unit: str = "ms"


class IncidentPayload(BaseModel):
    incident_id: str
    service_name: str
    environment: Literal["production", "staging", "edge"] = "production"
    title: str
    summary: str
    metric_name: str
    baseline_mean: float
    baseline_std: float = Field(gt=0.0)
    telemetry: List[TelemetryPoint] = Field(min_length=1)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    logs: List[str] = Field(default_factory=list)
    requested_by: str = "alertmanager"


class AnomalySignal(BaseModel):
    metric_name: str
    baseline_mean: float
    baseline_std: float
    observed_peak: float
    observed_mean: float
    z_score_peak: float
    ewma_latest: float
    ewma_drift_pct: float
    error_rate: float


class TelemetryReport(BaseModel):
    severity: Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
    is_anomalous: bool
    signal: AnomalySignal
    extracted_keywords: List[str]
    latency_ms: float


class RunbookChunk(BaseModel):
    runbook_id: str
    title: str
    domain: str
    keywords: List[str]
    content: str
    recommended_tool: str
    default_args: Dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    runbook_id: str
    title: str
    domain: str
    bm25_score: float
    cosine_score: float
    rrf_score: float
    bm25_rank: int
    cosine_rank: int
    snippet: str
    recommended_tool: str


class RAGContext(BaseModel):
    query: str
    hits: List[RetrievalHit]
    latency_ms: float


class ToolCallProposal(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str


class RCAReport(BaseModel):
    root_cause_hypothesis: str
    evidence_summary: str
    cited_runbooks: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_tools: List[ToolCallProposal] = Field(default_factory=list)
    reasoning_mode: Literal["gemini-llm", "deterministic-fallback"]
    latency_ms: float


class GuardrailCheck(BaseModel):
    check_name: str
    passed: bool
    detail: str


class GovernanceVerdict(BaseModel):
    decision: Literal[
        "APPROVED_AUTO",
        "REQUIRES_HUMAN_APPROVAL",
        "BLOCKED_SECURITY_VIOLATION",
    ]
    injection_detected: bool
    citation_grounded: bool
    blast_radius_score: float = Field(ge=0.0, le=1.0)
    checks: List[GuardrailCheck]
    reasons: List[str]
    latency_ms: float


class ToolExecutionResult(BaseModel):
    tool_name: str
    status: Literal["EXECUTED_SANDBOX", "SKIPPED_APPROVAL_REQUIRED", "BLOCKED"]
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: str


class EpisodicMemoryHit(BaseModel):
    trace_id: str
    incident_id: str
    service_name: str
    severity: str
    cited_runbook: str
    decision: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class CriticReflection(BaseModel):
    reflection_triggered: bool
    initial_rrf_score: float
    final_rrf_score: float
    rewritten_query: Optional[str] = None
    iterations: int = 1
    episodic_prior_used: bool = False
    critique_notes: str
    latency_ms: float


class IncidentResolutionTrace(BaseModel):
    trace_id: str
    created_at: str
    incident: IncidentPayload
    telemetry: TelemetryReport
    rag: RAGContext
    critic: CriticReflection
    episodic_memory: List[EpisodicMemoryHit] = Field(default_factory=list)
    rca: RCAReport
    governance: GovernanceVerdict
    executions: List[ToolExecutionResult] = Field(default_factory=list)
    total_latency_ms: float

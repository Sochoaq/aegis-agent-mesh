"""SQLite audit ledger for storing immutable incident resolution traces."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .schemas import EpisodicMemoryHit, IncidentResolutionTrace


class AuditLedger:
    """Persistent SQLite audit store for multi-agent traces and governance decisions."""

    def __init__(self, db_path: str | Path = "aegis_audit.db") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_traces (
                trace_id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                decision TEXT NOT NULL,
                injection_detected INTEGER NOT NULL,
                citation_grounded INTEGER NOT NULL,
                blast_radius_score REAL NOT NULL,
                total_latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def record_trace(self, trace: IncidentResolutionTrace) -> None:
        """Persist a full Pydantic IncidentResolutionTrace JSON document."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO incident_traces (
                trace_id, incident_id, service_name, severity, decision,
                injection_detected, citation_grounded, blast_radius_score,
                total_latency_ms, created_at, trace_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.incident.incident_id,
                trace.incident.service_name,
                trace.telemetry.severity,
                trace.governance.decision,
                1 if trace.governance.injection_detected else 0,
                1 if trace.governance.citation_grounded else 0,
                trace.governance.blast_radius_score,
                trace.total_latency_ms,
                trace.created_at,
                trace.model_dump_json(),
            ),
        )
        self.conn.commit()

    def list_traces(self, limit: int = 50) -> List[IncidentResolutionTrace]:
        """Return most recent incident traces ordered newest-first."""
        cursor = self.conn.execute(
            "SELECT trace_json FROM incident_traces ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )
        return [
            IncidentResolutionTrace.model_validate_json(row[0])
            for row in cursor.fetchall()
        ]

    def get_trace(self, trace_id: str) -> Optional[IncidentResolutionTrace]:
        row = self.conn.execute(
            "SELECT trace_json FROM incident_traces WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        return IncidentResolutionTrace.model_validate_json(row[0]) if row else None

    def find_similar_incidents(
        self,
        service_name: str,
        metric_name: str,
        keywords: List[str],
        exclude_incident_id: str = "",
        limit: int = 3,
    ) -> List[EpisodicMemoryHit]:
        """Retrieve non-blocked historical traces matching service, metric, and keywords."""
        cursor = self.conn.execute(
            """
            SELECT trace_json FROM incident_traces
            WHERE decision != 'BLOCKED_SECURITY_VIOLATION'
            ORDER BY created_at DESC LIMIT 100
            """
        )
        kw_set = set(keywords)
        scored: List[EpisodicMemoryHit] = []
        for (raw_json,) in cursor.fetchall():
            trace = IncidentResolutionTrace.model_validate_json(raw_json)
            if exclude_incident_id and trace.incident.incident_id == exclude_incident_id:
                continue

            svc_match = 1.0 if trace.incident.service_name == service_name else 0.0
            met_match = 1.0 if trace.incident.metric_name == metric_name else 0.0
            hist_kw = set(trace.telemetry.extracted_keywords)
            union = len(kw_set | hist_kw)
            jaccard = (len(kw_set & hist_kw) / union) if union > 0 else 0.0
            sim = round(0.40 * svc_match + 0.30 * met_match + 0.30 * jaccard, 4)

            if sim >= 0.25:
                cited = (
                    trace.rca.cited_runbooks[0]
                    if trace.rca.cited_runbooks
                    else (trace.rag.hits[0].runbook_id if trace.rag.hits else "")
                )
                scored.append(
                    EpisodicMemoryHit(
                        trace_id=trace.trace_id,
                        incident_id=trace.incident.incident_id,
                        service_name=trace.incident.service_name,
                        severity=trace.telemetry.severity,
                        cited_runbook=cited,
                        decision=trace.governance.decision,
                        similarity_score=sim,
                    )
                )

        scored.sort(key=lambda h: h.similarity_score, reverse=True)
        return scored[:limit]

    def close(self) -> None:
        self.conn.close()

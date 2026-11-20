import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from aegis.api import create_handler
from aegis.orchestrator import AegisOrchestrator
from aegis.schemas import IncidentPayload, TelemetryPoint
from aegis.storage import AuditLedger


class TestOrchestratorAndAPI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_audit.db"
        self.ledger = AuditLedger(self.db_path)
        self.orchestrator = AegisOrchestrator(ledger=self.ledger, use_llm=False)

    def tearDown(self):
        self.ledger.close()
        self.tmpdir.cleanup()

    def test_end_to_end_triage_and_sqlite_persistence(self):
        incident = IncidentPayload(
            incident_id="E2E-101",
            service_name="billing-postgres-proxy",
            environment="production",
            title="PgBouncer pool exhaustion",
            summary="SQL lock contention increasing pool wait latency.",
            metric_name="db_pool_wait_ms",
            baseline_mean=10.0,
            baseline_std=2.5,
            telemetry=[
                TelemetryPoint(timestamp="2026-11-20T11:00:00Z", value=12.0),
                TelemetryPoint(timestamp="2026-11-20T11:01:00Z", value=18.0),
            ],
            error_rate=0.05,
            logs=["WARN pgbouncer: pool_size exhausted waiting for idle connection"],
        )

        trace = self.orchestrator.run_incident(incident)
        self.assertEqual(trace.telemetry.severity, "SEV-2")
        self.assertEqual(trace.rag.hits[0].runbook_id, "RB-101")
        self.assertEqual(trace.governance.decision, "APPROVED_AUTO")
        self.assertEqual(trace.executions[0].status, "EXECUTED_SANDBOX")

        fetched = self.ledger.get_trace(trace.trace_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.incident.incident_id, "E2E-101")

    def test_http_api_health_and_triage_roundtrip(self):
        report_path = Path(self.tmpdir.name) / "latest_eval_report.json"
        handler = create_handler(self.orchestrator, report_path)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                health = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")

            payload = {
                "incident_id": "API-01",
                "service_name": "fraud-pytorch-scorer",
                "environment": "production",
                "title": "Model serving PSI drift breach",
                "summary": "Feature skew alert on inference endpoint.",
                "metric_name": "feature_psi_score",
                "baseline_mean": 0.08,
                "baseline_std": 0.02,
                "telemetry": [{"timestamp": "2026-11-20T11:00:00Z", "value": 0.14}],
                "error_rate": 0.01,
                "logs": ["WARN mlops: PSI drift > 0.25 on pytorch serving endpoint"],
            }
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/incidents/triage",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(result["rca"]["cited_runbooks"], ["RB-106"])
                self.assertEqual(result["governance"]["decision"], "APPROVED_AUTO")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

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

    def test_end_to_end_triage_episodic_memory_and_critic_reflection(self):
        # 1. Record first operational incident on billing-postgres-proxy
        first = IncidentPayload(
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

        trace1 = self.orchestrator.run_incident(first)
        self.assertEqual(trace1.telemetry.severity, "SEV-2")
        self.assertEqual(trace1.rag.hits[0].runbook_id, "RB-101")
        self.assertFalse(trace1.critic.reflection_triggered)
        self.assertEqual(trace1.critic.iterations, 1)
        self.assertEqual(trace1.governance.decision, "APPROVED_AUTO")
        self.assertEqual(trace1.executions[0].status, "EXECUTED_SANDBOX")

        fetched = self.ledger.get_trace(trace1.trace_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.incident.incident_id, "E2E-101")

        # 2. Terse follow-up on same service should hit Episodic Memory AND trigger Critic Reflection
        followup = IncidentPayload(
            incident_id="E2E-102",
            service_name="billing-postgres-proxy",
            environment="production",
            title="billing service slow queries and db hang",
            summary="Transactions stalling on proxy.",
            metric_name="db_pool_wait_ms",
            baseline_mean=10.0,
            baseline_std=2.5,
            telemetry=[TelemetryPoint(timestamp="2026-11-20T11:05:00Z", value=17.5)],
            error_rate=0.05,
            logs=["WARN billing: slow queries and db hang reported"],
        )
        trace2 = self.orchestrator.run_incident(followup)
        self.assertGreaterEqual(len(trace2.episodic_memory), 1)
        self.assertEqual(trace2.episodic_memory[0].incident_id, "E2E-101")
        self.assertTrue(trace2.critic.reflection_triggered)
        self.assertEqual(trace2.critic.iterations, 2)
        self.assertTrue(trace2.critic.episodic_prior_used)
        self.assertEqual(trace2.rag.hits[0].runbook_id, "RB-101")

    def test_http_api_console_openapi_and_triage_roundtrip(self):
        report_path = Path(self.tmpdir.name) / "latest_eval_report.json"
        handler = create_handler(self.orchestrator, report_path)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
                html_body = resp.read().decode("utf-8")
                self.assertIn("Aegis Agent Mesh", html_body)
                self.assertIn("/v1/incidents/triage", html_body)

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                health = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/scenarios", timeout=5) as resp:
                scenarios = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(len(scenarios), 15)

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=5) as resp:
                spec = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(spec["openapi"], "3.1.0")
                self.assertIn("/v1/incidents/triage", spec["paths"])

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
                self.assertIn("critic", result)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

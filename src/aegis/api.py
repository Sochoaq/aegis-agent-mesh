"""Zero-dependency HTTP JSON API exposing Aegis Multi-Agent Triage and Audit endpoints."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .orchestrator import AegisOrchestrator
from .schemas import IncidentPayload
from .storage import AuditLedger


def create_handler(orchestrator: AegisOrchestrator, report_path: Path):
    class AegisRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "service": "aegis-agent-mesh"})
                return

            if self.path == "/v1/evals/latest":
                if not report_path.exists():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "No benchmark report generated yet."})
                    return
                self._send_json(HTTPStatus.OK, json.loads(report_path.read_text(encoding="utf-8")))
                return

            if self.path == "/v1/audit/traces":
                if orchestrator.ledger is None:
                    self._send_json(HTTPStatus.OK, [])
                    return
                traces = [t.model_dump() for t in orchestrator.ledger.list_traces(limit=50)]
                self._send_json(HTTPStatus.OK, traces)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/incidents/triage":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
                incident = IncidentPayload.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            trace = orchestrator.run_incident(incident)
            self._send_json(HTTPStatus.OK, trace.model_dump())

        def log_message(self, format: str, *args) -> None:
            return

    return AegisRequestHandler


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    db_path: Optional[Path] = None,
) -> None:
    root = Path(__file__).resolve().parents[2]
    ledger = AuditLedger(db_path or (root / "aegis_audit.db"))
    orch = AegisOrchestrator(ledger=ledger)
    handler = create_handler(orch, root / "benchmarks" / "latest_eval_report.json")
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Aegis API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        ledger.close()

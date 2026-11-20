"""Self-contained dark-mode HTML observability & LLMOps governance dashboard generator."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List

from .schemas import IncidentResolutionTrace


def generate_html_dashboard(
    eval_report: Dict[str, Any],
    traces: List[IncidentResolutionTrace],
    output_path: Path,
) -> Path:
    """Render an offline dark-mode HTML dashboard summarizing benchmark KPIs and audit traces."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_html: List[str] = []
    for item in eval_report.get("scenarios", []):
        decision = str(item.get("decision", ""))
        badge_cls = "badge-ok"
        if decision == "REQUIRES_HUMAN_APPROVAL":
            badge_cls = "badge-warn"
        elif decision == "BLOCKED_SECURITY_VIOLATION":
            badge_cls = "badge-danger"

        rows_html.append(
            f"""
            <tr>
              <td><strong>{html.escape(str(item.get('scenario_id')))}</strong></td>
              <td>{html.escape(str(item.get('category')))}</td>
              <td>{html.escape(str(item.get('service')))}</td>
              <td>{html.escape(str(item.get('severity')))}</td>
              <td>{item.get('z_score_peak')}</td>
              <td><code>{html.escape(str(item.get('top1_runbook')))}</code></td>
              <td>{item.get('critic_iterations', 1)}</td>
              <td>{item.get('episodic_hits', 0)}</td>
              <td><span class="badge {badge_cls}">{html.escape(decision)}</span></td>
              <td>{item.get('blast_radius_score')}</td>
              <td>{item.get('latency_ms')} ms</td>
            </tr>
            """
        )

    trace_count = len(traces)
    top1_pct = round(float(eval_report.get("rca_runbook_top1_accuracy", 0.0)) * 100.0, 1)
    ground_pct = round(float(eval_report.get("grounding_citation_rate", 0.0)) * 100.0, 1)
    adv_pct = round(float(eval_report.get("adversarial_injection_block_rate", 0.0)) * 100.0, 1)
    p95_ms = eval_report.get("p95_latency_ms", 0.0)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Aegis Agent Mesh — LLMOps Observability Dashboard</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --ok: #3fb950;
      --warn: #d29922;
      --danger: #f85149;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      padding: 2rem;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.25rem;
      margin-bottom: 1.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }}
    h1 {{ font-size: 1.6rem; font-weight: 600; }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 1rem;
      margin-bottom: 1.75rem;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.25rem;
    }}
    .label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }}
    .val {{ font-size: 1.8rem; font-weight: 600; margin-top: 0.35rem; }}
    .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow-x: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
    th {{
      background: #1c2128;
      color: var(--muted);
      font-weight: 600;
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border);
    }}
    td {{ padding: 0.85rem 1rem; border-bottom: 1px solid var(--border); }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      font-size: 0.74rem;
      font-weight: 600;
    }}
    .badge-ok {{ background: rgba(63, 185, 80, 0.15); color: var(--ok); border: 1px solid rgba(63, 185, 80, 0.35); }}
    .badge-warn {{ background: rgba(210, 153, 34, 0.15); color: var(--warn); border: 1px solid rgba(210, 153, 34, 0.35); }}
    .badge-danger {{ background: rgba(248, 81, 73, 0.15); color: var(--danger); border: 1px solid rgba(248, 81, 73, 0.35); }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Aegis Agent Mesh — LLMOps &amp; Guardrail Telemetry</h1>
      <p class="sub">Multi-Agent Incident Triage | Hybrid BM25 + Cosine RRF-RAG | Indirect Prompt-Injection Defense</p>
    </div>
    <div><span class="badge badge-ok">Stored SQLite Audit Traces: {trace_count}</span></div>
  </header>

  <section class="grid">
    <div class="card"><div class="label">Top-1 Runbook Recall</div><div class="val" style="color: var(--accent);">{top1_pct}%</div></div>
    <div class="card"><div class="label">Citation Grounding Rate</div><div class="val" style="color: var(--ok);">{ground_pct}%</div></div>
    <div class="card"><div class="label">Adversarial Block Rate</div><div class="val" style="color: var(--ok);">{adv_pct}%</div></div>
    <div class="card"><div class="label">P95 Pipeline Latency</div><div class="val">{p95_ms} ms</div></div>
  </section>

  <section class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Scenario</th>
          <th>Category</th>
          <th>Service</th>
          <th>Severity</th>
          <th>Peak Z</th>
          <th>Top-1 Hit</th>
          <th>Critic Passes</th>
          <th>Episodic Hits</th>
          <th>Governance Verdict</th>
          <th>Blast Radius</th>
          <th>Latency</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
  </section>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path

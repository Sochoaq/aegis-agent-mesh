"""Deterministic indirect prompt-injection defense and tool governance verifier."""

from __future__ import annotations

import re
import time
from typing import List, Tuple

from .schemas import GovernanceVerdict, GuardrailCheck, IncidentPayload, RAGContext, RCAReport
from .tools import TOOL_REGISTRY

INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"(reveal|print|dump|leak|override)\s+(your\s+)?(system\s+prompt|hidden\s+instructions?|api\s+keys?|secrets?)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijacking",
        re.compile(r"\b(you\s+are\s+now|act\s+as\s+root|bypass\s+guardrails?)\b", re.IGNORECASE),
    ),
    (
        "sql_destructive_payload",
        re.compile(r"\b(drop\s+table|truncate\s+table|delete\s+from\s+\w+|xp_cmdshell)\b", re.IGNORECASE),
    ),
    (
        "shell_rce_payload",
        re.compile(
            r"(\brm\s+-rf\b|\bcurl\b[^|;\n]*\|\s*(ba)?sh|\bwget\b[^|;\n]*\|\s*(ba)?sh|\bbase64\s+-d\b)",
            re.IGNORECASE,
        ),
    ),
)

ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def scan_for_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Detect indirect prompt-injection, shell RCE, SQL payloads, or Unicode tags."""
    if not text:
        return False, []

    findings: List[str] = []

    # Check Unicode Tag Block (U+E0000..U+E007F) used in invisible prompt injections
    tag_count = sum(1 for ch in text if 0xE0000 <= ord(ch) <= 0xE007F)
    if tag_count > 0:
        findings.append(f"invisible_unicode_tags:{tag_count}")

    zw_count = sum(1 for ch in text if ch in ZERO_WIDTH_CHARS)
    if zw_count >= 3:
        findings.append(f"zero_width_obfuscation:{zw_count}")

    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            findings.append(name)

    return len(findings) > 0, findings


def verify_tool_policy_and_grounding(
    incident: IncidentPayload,
    rag_context: RAGContext,
    rca_report: RCAReport,
    severity: str = "SEV-2",
) -> GovernanceVerdict:
    """Verify prompt integrity, runbook citation grounding, and blast-radius policy."""
    t0 = time.perf_counter()
    checks: List[GuardrailCheck] = []
    reasons: List[str] = []

    # 1. Indirect Prompt-Injection Scan across all untrusted text inputs
    untrusted_blobs = [incident.title, incident.summary, *incident.logs]
    injection_findings: List[str] = []
    for blob in untrusted_blobs:
        detected, hits = scan_for_prompt_injection(blob)
        if detected:
            injection_findings.extend(hits)

    injection_detected = len(injection_findings) > 0
    checks.append(
        GuardrailCheck(
            check_name="indirect_prompt_injection_scan",
            passed=not injection_detected,
            detail=(
                "Clean telemetry & log stream"
                if not injection_detected
                else f"Adversarial patterns detected: {sorted(set(injection_findings))}"
            ),
        )
    )
    if injection_detected:
        reasons.append(
            f"Blocked due to indirect prompt injection / payload markers: {sorted(set(injection_findings))}"
        )

    # 2. Citation Grounding Verification ([RB-xxx] must be non-empty and in retrieved hits)
    retrieved_ids = {hit.runbook_id for hit in rag_context.hits}
    cited_ids = list(dict.fromkeys(rca_report.cited_runbooks))
    invalid_citations = [cid for cid in cited_ids if cid not in retrieved_ids]
    citation_grounded = len(cited_ids) > 0 and len(invalid_citations) == 0

    checks.append(
        GuardrailCheck(
            check_name="rag_citation_grounding",
            passed=citation_grounded,
            detail=(
                f"Grounded in retrieved runbooks: {cited_ids}"
                if citation_grounded
                else f"Ungrounded or missing citations (cited={cited_ids}, retrieved={sorted(retrieved_ids)})"
            ),
        )
    )
    if not citation_grounded:
        reasons.append("RCA failed runbook citation grounding verification.")

    # 3. Tool Registry, Argument Validation & Blast-Radius Assessment
    max_blast_radius = 0.0
    tool_policy_passed = True
    requires_human_approval = False

    for proposal in rca_report.proposed_tools:
        spec = TOOL_REGISTRY.get(proposal.tool_name)
        if spec is None:
            tool_policy_passed = False
            max_blast_radius = 1.0
            reasons.append(f"Unknown or unapproved tool requested: '{proposal.tool_name}'")
            continue

        valid_args, arg_detail = spec.validator(proposal.arguments)
        if not valid_args:
            tool_policy_passed = False
            max_blast_radius = max(max_blast_radius, 1.0)
            reasons.append(f"Unsafe parameters for '{proposal.tool_name}': {arg_detail}")
            continue

        max_blast_radius = max(max_blast_radius, spec.blast_radius_score)
        if spec.risk_tier in ("HIGH", "CRITICAL"):
            requires_human_approval = True
            reasons.append(
                f"Tool '{proposal.tool_name}' carries {spec.risk_tier} risk tier (blast_radius={spec.blast_radius_score:.2f}); requires operator sign-off."
            )

    if severity == "SEV-1" and rca_report.proposed_tools:
        requires_human_approval = True
        reasons.append("SEV-1 critical production incident requires human-in-the-loop sign-off.")

    checks.append(
        GuardrailCheck(
            check_name="tool_registry_and_parameter_safety",
            passed=tool_policy_passed,
            detail=(
                f"All proposed tools validated (max_blast_radius={max_blast_radius:.2f})"
                if tool_policy_passed
                else "Tool allowlist or argument validation violation"
            ),
        )
    )

    if injection_detected or not tool_policy_passed or not citation_grounded:
        decision = "BLOCKED_SECURITY_VIOLATION"
    elif requires_human_approval:
        decision = "REQUIRES_HUMAN_APPROVAL"
    else:
        decision = "APPROVED_AUTO"
        reasons.append("All guardrail checks passed within automated blast-radius budget.")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return GovernanceVerdict(
        decision=decision,
        injection_detected=injection_detected,
        citation_grounded=citation_grounded,
        blast_radius_score=round(max_blast_radius, 2),
        checks=checks,
        reasons=reasons,
        latency_ms=elapsed_ms,
    )

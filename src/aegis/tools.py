"""Sandboxed remediation tool registry, blast-radius tiers, and dry-run executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from .schemas import ToolCallProposal, ToolExecutionResult

BLAST_RADIUS_SCORES: Dict[str, float] = {
    "LOW": 0.20,
    "MEDIUM": 0.50,
    "HIGH": 0.85,
    "CRITICAL": 1.00,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    risk_tier: str
    description: str
    required_args: Tuple[str, ...]
    validator: Callable[[Dict[str, Any]], Tuple[bool, str]]

    @property
    def blast_radius_score(self) -> float:
        return BLAST_RADIUS_SCORES[self.risk_tier]


def _validate_nonempty_str(args: Dict[str, Any], key: str) -> Tuple[bool, str]:
    val = args.get(key)
    if not isinstance(val, str) or not val.strip():
        return False, f"Argument '{key}' must be a non-empty string."
    if any(tok in val for tok in (";", "&&", "|", "`", "$(", "--")):
        return False, f"Argument '{key}' contains forbidden shell/SQL characters."
    return True, "OK"


def _validate_drain_pool(args: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = _validate_nonempty_str(args, "service")
    if not ok:
        return False, msg
    timeout = args.get("idle_timeout_sec", 30)
    if not isinstance(timeout, int) or not (5 <= timeout <= 300):
        return False, "Argument 'idle_timeout_sec' must be an int in [5, 300]."
    return True, "OK"


def _validate_scale_replicas(args: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = _validate_nonempty_str(args, "service")
    if not ok:
        return False, msg
    replicas = args.get("target_replicas")
    if not isinstance(replicas, int) or not (1 <= replicas <= 12):
        return False, "Argument 'target_replicas' must be an integer between 1 and 12."
    return True, "OK"


def _validate_rollback(args: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = _validate_nonempty_str(args, "service")
    if not ok:
        return False, msg
    return _validate_nonempty_str(args, "target_revision")


def _validate_isolate_gateway(args: Dict[str, Any]) -> Tuple[bool, str]:
    return _validate_nonempty_str(args, "gateway_id")


def _validate_rotate_creds(args: Dict[str, Any]) -> Tuple[bool, str]:
    return _validate_nonempty_str(args, "principal")


def _validate_recalibrate_model(args: Dict[str, Any]) -> Tuple[bool, str]:
    return _validate_nonempty_str(args, "model_name")


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "drain_db_connection_pool": ToolSpec(
        name="drain_db_connection_pool",
        risk_tier="LOW",
        description="Recycle idle-in-transaction pooled connections cleanly without dropping active queries.",
        required_args=("service",),
        validator=_validate_drain_pool,
    ),
    "trigger_model_recalibration": ToolSpec(
        name="trigger_model_recalibration",
        risk_tier="LOW",
        description="Schedule asynchronous feature baseline recalibration and shadow validation pipeline.",
        required_args=("model_name",),
        validator=_validate_recalibrate_model,
    ),
    "scale_service_replicas": ToolSpec(
        name="scale_service_replicas",
        risk_tier="MEDIUM",
        description="Horizontally adjust stateless worker replicas within bounded safety limits.",
        required_args=("service", "target_replicas"),
        validator=_validate_scale_replicas,
    ),
    "rollback_deployment": ToolSpec(
        name="rollback_deployment",
        risk_tier="HIGH",
        description="Revert production Kubernetes deployment to last verified stable revision.",
        required_args=("service", "target_revision"),
        validator=_validate_rollback,
    ),
    "isolate_edge_gateway": ToolSpec(
        name="isolate_edge_gateway",
        risk_tier="HIGH",
        description="Quarantine noisy/drifting IoT/BLE edge telemetry gateway to backup broker.",
        required_args=("gateway_id",),
        validator=_validate_isolate_gateway,
    ),
    "rotate_api_credentials": ToolSpec(
        name="rotate_api_credentials",
        risk_tier="CRITICAL",
        description="Revoke and reissue compromised IAM/API signing keys across active principals.",
        required_args=("principal",),
        validator=_validate_rotate_creds,
    ),
}


def execute_sandboxed_tool(
    proposal: ToolCallProposal, governance_decision: str
) -> ToolExecutionResult:
    """Simulate execution in a deterministic sandbox according to governance policy."""
    spec = TOOL_REGISTRY.get(proposal.tool_name)
    if spec is None or governance_decision == "BLOCKED_SECURITY_VIOLATION":
        return ToolExecutionResult(
            tool_name=proposal.tool_name,
            status="BLOCKED",
            arguments=proposal.arguments,
            output="Execution refused by PolicyGuardrailAgent.",
        )

    valid, detail = spec.validator(proposal.arguments)
    if not valid:
        return ToolExecutionResult(
            tool_name=proposal.tool_name,
            status="BLOCKED",
            arguments=proposal.arguments,
            output=f"Parameter validation failed: {detail}",
        )

    if governance_decision == "REQUIRES_HUMAN_APPROVAL":
        return ToolExecutionResult(
            tool_name=proposal.tool_name,
            status="SKIPPED_APPROVAL_REQUIRED",
            arguments=proposal.arguments,
            output=(
                f"Queued for operator sign-off (risk={spec.risk_tier}, "
                f"blast_radius={spec.blast_radius_score:.2f})."
            ),
        )

    return ToolExecutionResult(
        tool_name=proposal.tool_name,
        status="EXECUTED_SANDBOX",
        arguments=proposal.arguments,
        output=(
            f"[SANDBOX] Executed '{proposal.tool_name}' successfully with "
            f"args={proposal.arguments}."
        ),
    )

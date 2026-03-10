"""
STC Framework — Prompt Logging & Auditability
observability/prompt_logger.py

Complete prompt lifecycle tracking for regulated environments:
  - Full prompt/response capture (template + rendered + response)
  - Prompt version management with diff tracking
  - Token and cost attribution per prompt execution
  - Prompt performance analytics (accuracy, hallucination, latency by version)
  - Regulatory query support ("show me the exact prompt for request X")
  - A/B testing framework for prompt comparison
  - PII-safe logging (prompts are logged AFTER Presidio redaction)
  - Langfuse integration bridge for enterprise deployments

Answers the auditor's question: "Show me the exact prompt sent to the LLM
for this specific query on March 5th, and how that prompt has changed
over the past 6 months."

Part of the Observability layer. Integrates with:
  - Data Lineage (lineage_id correlation)
  - Audit Trail (immutable event logging)
  - Cost Controls (per-prompt cost attribution)
  - Critic (prompt → verdict correlation)
"""

import json
import hashlib
import logging
import time
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("stc.observability.prompt_logger")


# ── Prompt Template Management ──────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """A versioned prompt template with metadata."""
    template_id: str
    version: str
    template_text: str           # The template with {variable} placeholders
    system_prompt: str = ""      # System prompt (if separate)
    purpose: str = ""
    author: str = ""
    created_at: str = ""
    variables: List[str] = field(default_factory=list)  # Expected variables
    model_constraints: List[str] = field(default_factory=list)  # Which models this works with
    tags: List[str] = field(default_factory=list)
    status: str = "active"       # draft | review | active | deprecated
    
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            f"{self.system_prompt}||{self.template_text}".encode()
        ).hexdigest()[:16]

    def render(self, variables: Dict[str, str]) -> str:
        """Render the template with provided variables."""
        rendered = self.template_text
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id, "version": self.version,
            "template_text": self.template_text, "system_prompt": self.system_prompt,
            "purpose": self.purpose, "author": self.author,
            "created_at": self.created_at, "content_hash": self.content_hash,
            "variables": self.variables, "status": self.status, "tags": self.tags,
        }


@dataclass
class PromptVersion:
    """Tracks a version change with diff information."""
    template_id: str
    from_version: str
    to_version: str
    changed_at: str
    changed_by: str
    change_type: str    # created | modified | deprecated | restored
    diff_summary: str   # Human-readable diff
    from_hash: str = ""
    to_hash: str = ""


# ── Prompt Execution Log ────────────────────────────────────────────────────

@dataclass
class PromptExecution:
    """
    Complete record of a single prompt execution.
    This is the auditable artifact — the exact input/output pair.
    """
    execution_id: str
    timestamp: str
    # Correlation IDs
    lineage_id: str = ""         # Links to data lineage
    session_id: str = ""
    request_id: str = ""
    # Template info
    template_id: str = ""
    template_version: str = ""
    template_hash: str = ""
    # Actual content (post-PII-redaction)
    system_prompt: str = ""      # Rendered system prompt
    user_prompt: str = ""        # Rendered user prompt (after PII masking)
    full_prompt: str = ""        # Complete prompt as sent to LLM
    # Context
    context_documents: List[str] = field(default_factory=list)
    context_tokens: int = 0
    variables_used: Dict[str, str] = field(default_factory=dict)
    # LLM details
    model_id: str = ""
    provider: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    # Response
    response_text: str = ""
    response_tokens: int = 0
    response_hash: str = ""
    # Performance
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    # Governance
    critic_verdict: str = ""     # pass | fail | escalate
    critic_violations: List[str] = field(default_factory=list)
    pii_redactions: int = 0
    data_tier: str = "public"
    # Status
    status: str = "completed"    # completed | failed | timeout | blocked

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @property
    def cost_breakdown(self) -> Dict[str, float]:
        # Approximate cost calculation (rates vary by model)
        rates = {
            "claude-sonnet-4": {"input": 0.003, "output": 0.015},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "llama-3.1-8b": {"input": 0.0, "output": 0.0},
        }
        model_short = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
        rate = rates.get(model_short, {"input": 0.003, "output": 0.015})
        input_cost = self.prompt_tokens / 1000 * rate["input"]
        output_cost = self.completion_tokens / 1000 * rate["output"]
        return {"input_cost": round(input_cost, 6), "output_cost": round(output_cost, 6),
                "total_cost": round(input_cost + output_cost, 6)}


# ── Prompt Registry ─────────────────────────────────────────────────────────

class PromptRegistry:
    """
    Manages prompt templates with full version history.
    In production, backed by Langfuse. This module provides the
    local implementation + Langfuse bridge interface.
    """

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}  # template_id → current
        self._version_history: Dict[str, List[PromptTemplate]] = defaultdict(list)
        self._changes: List[PromptVersion] = []

    def register(self, template: PromptTemplate) -> PromptTemplate:
        """Register a new template or new version of existing template."""
        now = datetime.now(timezone.utc).isoformat()
        template.created_at = template.created_at or now

        old = self._templates.get(template.template_id)
        change_type = "modified" if old else "created"
        from_version = old.version if old else ""
        from_hash = old.content_hash if old else ""

        self._templates[template.template_id] = template
        self._version_history[template.template_id].append(copy.deepcopy(template))

        diff = self._compute_diff(old, template) if old else "Initial version"
        self._changes.append(PromptVersion(
            template_id=template.template_id,
            from_version=from_version, to_version=template.version,
            changed_at=now, changed_by=template.author,
            change_type=change_type, diff_summary=diff,
            from_hash=from_hash, to_hash=template.content_hash,
        ))

        return template

    def get(self, template_id: str) -> Optional[PromptTemplate]:
        return self._templates.get(template_id)

    def get_version(self, template_id: str, version: str) -> Optional[PromptTemplate]:
        for t in self._version_history.get(template_id, []):
            if t.version == version:
                return t
        return None

    def list_versions(self, template_id: str) -> List[Dict[str, Any]]:
        return [
            {"version": t.version, "hash": t.content_hash, "created": t.created_at,
             "author": t.author, "status": t.status}
            for t in self._version_history.get(template_id, [])
        ]

    def deprecate(self, template_id: str, by: str = "system"):
        t = self._templates.get(template_id)
        if t:
            t.status = "deprecated"
            self._changes.append(PromptVersion(
                template_id=template_id, from_version=t.version, to_version=t.version,
                changed_at=datetime.now(timezone.utc).isoformat(), changed_by=by,
                change_type="deprecated", diff_summary="Template deprecated",
            ))

    def change_log(self, template_id: Optional[str] = None) -> List[Dict[str, Any]]:
        changes = self._changes
        if template_id:
            changes = [c for c in changes if c.template_id == template_id]
        return [c.__dict__ for c in changes]

    def _compute_diff(self, old: PromptTemplate, new: PromptTemplate) -> str:
        diffs = []
        if old.system_prompt != new.system_prompt:
            diffs.append("System prompt modified")
        if old.template_text != new.template_text:
            old_lines = old.template_text.split("\n")
            new_lines = new.template_text.split("\n")
            added = len(new_lines) - len(old_lines)
            diffs.append(f"Template body: {'+' if added >= 0 else ''}{added} lines")
        if old.variables != new.variables:
            diffs.append(f"Variables changed: {old.variables} → {new.variables}")
        return "; ".join(diffs) if diffs else "No content changes (metadata only)"


# ── Prompt Logger ───────────────────────────────────────────────────────────

class PromptLogger:
    """
    Central prompt logging engine. Captures every prompt execution
    with full content, metadata, and governance context.

    Usage:
        logger = PromptLogger(registry=registry)

        # Start an execution context
        ctx = logger.start_execution(
            template_id="financial_qa_v3",
            variables={"query": query, "context": context},
            session_id="abc", lineage_id="trace-123")

        # After LLM call
        logger.complete_execution(ctx.execution_id,
            response_text="...", model_id="claude-sonnet-4",
            prompt_tokens=2200, completion_tokens=350, latency_ms=2100,
            critic_verdict="pass")

        # Query
        history = logger.get_execution_history(template_id="financial_qa_v3")
        exact = logger.get_execution("exec-123")
    """

    def __init__(self, registry: Optional[PromptRegistry] = None,
                 audit_callback: Optional[Callable] = None,
                 max_response_log_chars: int = 5000):
        self.registry = registry or PromptRegistry()
        self._executions: Dict[str, PromptExecution] = {}
        self._by_template: Dict[str, List[str]] = defaultdict(list)
        self._by_session: Dict[str, List[str]] = defaultdict(list)
        self._by_lineage: Dict[str, List[str]] = defaultdict(list)
        self._audit_callback = audit_callback
        self._max_response_chars = max_response_log_chars
        self._counter = 0

    def _gen_id(self) -> str:
        self._counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"pex-{ts}-{self._counter:06d}"

    def start_execution(self, template_id: str, variables: Dict[str, str],
                        session_id: str = "", lineage_id: str = "",
                        request_id: str = "", data_tier: str = "public",
                        pii_redactions: int = 0,
                        context_documents: List[str] = None) -> PromptExecution:
        """
        Start tracking a prompt execution. Call before the LLM call.
        Returns the execution record to be completed after the LLM responds.
        """
        now = datetime.now(timezone.utc).isoformat()
        exec_id = self._gen_id()

        template = self.registry.get(template_id)
        if not template:
            # Create a minimal template record for unregistered prompts
            template = PromptTemplate(
                template_id=template_id, version="unregistered",
                template_text=str(variables.get("prompt", "")),
            )

        # Render the prompt
        rendered = template.render(variables)
        full_prompt = f"{template.system_prompt}\n\n{rendered}".strip() if template.system_prompt else rendered

        execution = PromptExecution(
            execution_id=exec_id, timestamp=now,
            lineage_id=lineage_id, session_id=session_id, request_id=request_id,
            template_id=template_id, template_version=template.version,
            template_hash=template.content_hash,
            system_prompt=template.system_prompt,
            user_prompt=rendered,
            full_prompt=full_prompt,
            context_documents=context_documents or [],
            context_tokens=len(full_prompt.split()) * 4 // 3,  # rough token estimate
            variables_used={k: v[:200] for k, v in variables.items()},  # truncate large values
            pii_redactions=pii_redactions,
            data_tier=data_tier,
            status="in_progress",
        )

        self._executions[exec_id] = execution
        self._by_template[template_id].append(exec_id)
        if session_id:
            self._by_session[session_id].append(exec_id)
        if lineage_id:
            self._by_lineage[lineage_id].append(exec_id)

        return execution

    def complete_execution(self, execution_id: str,
                           response_text: str, model_id: str, provider: str = "",
                           prompt_tokens: int = 0, completion_tokens: int = 0,
                           latency_ms: float = 0.0, temperature: float = 0.0,
                           critic_verdict: str = "", critic_violations: List[str] = None,
                           estimated_cost: float = 0.0) -> PromptExecution:
        """Complete an execution record after the LLM responds."""
        exe = self._executions.get(execution_id)
        if not exe:
            raise KeyError(f"Execution not found: {execution_id}")

        # Truncate response for storage if needed
        logged_response = response_text[:self._max_response_chars]
        if len(response_text) > self._max_response_chars:
            logged_response += f"\n[...truncated, {len(response_text)} total chars]"

        exe.response_text = logged_response
        exe.response_hash = hashlib.sha256(response_text.encode()).hexdigest()[:16]
        exe.response_tokens = completion_tokens
        exe.model_id = model_id
        exe.provider = provider
        exe.temperature = temperature
        exe.prompt_tokens = prompt_tokens
        exe.completion_tokens = completion_tokens
        exe.total_tokens = prompt_tokens + completion_tokens
        exe.latency_ms = latency_ms
        exe.estimated_cost = estimated_cost or exe.cost_breakdown["total_cost"]
        exe.critic_verdict = critic_verdict
        exe.critic_violations = critic_violations or []
        exe.status = "completed"

        self._emit_audit("prompt_execution_completed", exe)
        return exe

    def fail_execution(self, execution_id: str, error: str):
        """Mark an execution as failed."""
        exe = self._executions.get(execution_id)
        if exe:
            exe.status = "failed"
            exe.response_text = f"ERROR: {error}"
            self._emit_audit("prompt_execution_failed", exe)

    # ── Query Interface ─────────────────────────────────────────────────

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a single execution record — the auditor's primary query."""
        exe = self._executions.get(execution_id)
        return exe.to_dict() if exe else None

    def get_by_lineage(self, lineage_id: str) -> List[Dict[str, Any]]:
        """Get all prompt executions for a data lineage trace."""
        ids = self._by_lineage.get(lineage_id, [])
        return [self._executions[eid].to_dict() for eid in ids if eid in self._executions]

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all prompt executions for a user session."""
        ids = self._by_session.get(session_id, [])
        return [self._executions[eid].to_dict() for eid in ids if eid in self._executions]

    def get_by_template(self, template_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history for a specific template."""
        ids = self._by_template.get(template_id, [])[-limit:]
        return [self._executions[eid].to_dict() for eid in ids if eid in self._executions]

    def get_by_date_range(self, start: str, end: str) -> List[Dict[str, Any]]:
        """Get all executions within a date range."""
        return [
            exe.to_dict() for exe in self._executions.values()
            if start <= exe.timestamp <= end
        ]

    # ── Analytics ───────────────────────────────────────────────────────

    def template_performance(self, template_id: str) -> Dict[str, Any]:
        """
        Analyze prompt template performance across all executions.
        Used for A/B testing and version comparison.
        """
        ids = self._by_template.get(template_id, [])
        exes = [self._executions[eid] for eid in ids if eid in self._executions]

        if not exes:
            return {"template_id": template_id, "total_executions": 0}

        completed = [e for e in exes if e.status == "completed"]
        by_version = defaultdict(list)
        for e in completed:
            by_version[e.template_version].append(e)

        version_stats = {}
        for ver, ver_exes in by_version.items():
            latencies = [e.latency_ms for e in ver_exes]
            costs = [e.estimated_cost for e in ver_exes]
            verdicts = [e.critic_verdict for e in ver_exes]
            pass_rate = sum(1 for v in verdicts if v == "pass") / len(verdicts) if verdicts else 0

            version_stats[ver] = {
                "executions": len(ver_exes),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                "avg_cost": round(sum(costs) / len(costs), 6) if costs else 0,
                "total_cost": round(sum(costs), 4),
                "critic_pass_rate": round(pass_rate, 4),
                "avg_tokens": round(sum(e.total_tokens for e in ver_exes) / len(ver_exes)) if ver_exes else 0,
                "fail_count": sum(1 for v in verdicts if v == "fail"),
            }

        return {
            "template_id": template_id,
            "total_executions": len(exes),
            "completed": len(completed),
            "failed": len([e for e in exes if e.status == "failed"]),
            "versions": version_stats,
            "total_cost": round(sum(e.estimated_cost for e in completed), 4),
        }

    def compare_versions(self, template_id: str, version_a: str,
                         version_b: str) -> Dict[str, Any]:
        """A/B comparison between two prompt versions."""
        perf = self.template_performance(template_id)
        va = perf.get("versions", {}).get(version_a, {})
        vb = perf.get("versions", {}).get(version_b, {})

        if not va or not vb:
            return {"error": "One or both versions not found", "available": list(perf.get("versions", {}).keys())}

        def delta(metric):
            a_val = va.get(metric, 0)
            b_val = vb.get(metric, 0)
            if a_val == 0:
                return "N/A"
            return f"{((b_val - a_val) / a_val * 100):+.1f}%"

        return {
            "template_id": template_id,
            "version_a": version_a, "version_b": version_b,
            "comparison": {
                "executions": {"a": va["executions"], "b": vb["executions"]},
                "avg_latency_ms": {"a": va["avg_latency_ms"], "b": vb["avg_latency_ms"],
                                    "delta": delta("avg_latency_ms")},
                "avg_cost": {"a": va["avg_cost"], "b": vb["avg_cost"],
                             "delta": delta("avg_cost")},
                "critic_pass_rate": {"a": va["critic_pass_rate"], "b": vb["critic_pass_rate"],
                                     "delta": delta("critic_pass_rate")},
                "avg_tokens": {"a": va["avg_tokens"], "b": vb["avg_tokens"],
                               "delta": delta("avg_tokens")},
            },
            "recommendation": (
                f"Version {version_b} is better" if vb.get("critic_pass_rate", 0) >= va.get("critic_pass_rate", 0)
                and vb.get("avg_cost", 999) <= va.get("avg_cost", 0) * 1.1
                else f"Version {version_a} is better or inconclusive"
            ),
        }

    # ── Compliance Reporting ────────────────────────────────────────────

    def audit_report(self, start: str = "", end: str = "") -> Dict[str, Any]:
        """Generate a compliance-ready audit report of all prompt activity."""
        exes = list(self._executions.values())
        if start:
            exes = [e for e in exes if e.timestamp >= start]
        if end:
            exes = [e for e in exes if e.timestamp <= end]

        total = len(exes)
        completed = [e for e in exes if e.status == "completed"]
        failed = [e for e in exes if e.status == "failed"]
        blocked = [e for e in exes if e.status == "blocked"]

        by_model = defaultdict(int)
        by_template = defaultdict(int)
        by_tier = defaultdict(int)
        total_cost = 0.0
        total_tokens = 0
        pii_redaction_total = 0
        critic_verdicts = defaultdict(int)

        for e in completed:
            by_model[e.model_id] += 1
            by_template[e.template_id] += 1
            by_tier[e.data_tier] += 1
            total_cost += e.estimated_cost
            total_tokens += e.total_tokens
            pii_redaction_total += e.pii_redactions
            if e.critic_verdict:
                critic_verdicts[e.critic_verdict] += 1

        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "period": {"start": start or "all", "end": end or "all"},
            "total_executions": total,
            "completed": len(completed),
            "failed": len(failed),
            "blocked": len(blocked),
            "by_model": dict(by_model),
            "by_template": dict(by_template),
            "by_data_tier": dict(by_tier),
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "pii_redactions_total": pii_redaction_total,
            "critic_verdicts": dict(critic_verdicts),
            "templates_active": len(set(e.template_id for e in exes)),
            "unique_sessions": len(set(e.session_id for e in exes if e.session_id)),
        }

    def _emit_audit(self, event_type: str, exe: PromptExecution):
        if self._audit_callback:
            self._audit_callback({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": "observability.prompt_logger",
                "event_type": event_type,
                "details": {
                    "execution_id": exe.execution_id,
                    "template_id": exe.template_id,
                    "template_version": exe.template_version,
                    "model_id": exe.model_id,
                    "tokens": exe.total_tokens,
                    "cost": exe.estimated_cost,
                    "critic_verdict": exe.critic_verdict,
                    "status": exe.status,
                    "lineage_id": exe.lineage_id,
                },
            })


# ── Demo ────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 70)
    print("STC Prompt Logging & Auditability — Demo")
    print("=" * 70)

    audit_log = []
    registry = PromptRegistry()
    plogger = PromptLogger(registry=registry, audit_callback=lambda e: audit_log.append(e))

    # ── Register prompt templates ──
    print("\n▸ Registering prompt templates...")

    t1 = registry.register(PromptTemplate(
        template_id="financial_qa",
        version="3.0",
        system_prompt="You are a financial analyst assistant. Answer questions based ONLY on the provided documents. Never provide investment advice.",
        template_text="Based on the following documents:\n\n{context}\n\nAnswer the following question:\n{query}\n\nProvide specific numbers and cite your sources.",
        purpose="Financial document Q&A",
        author="ML Engineering",
        variables=["context", "query"],
        model_constraints=["claude-sonnet-4", "gpt-4o"],
        tags=["financial", "rag", "production"],
    ))
    print(f"  Registered: {t1.template_id} v{t1.version} (hash: {t1.content_hash})")

    # Register updated version
    t1v2 = registry.register(PromptTemplate(
        template_id="financial_qa",
        version="3.1",
        system_prompt="You are a financial analyst assistant for a wealth management firm. Answer questions based ONLY on the provided documents. Never provide investment advice or recommendations. If uncertain, say so explicitly.",
        template_text="Based on the following documents:\n\n{context}\n\nAnswer the following question:\n{query}\n\nProvide specific numbers, cite your sources, and indicate confidence level (high/medium/low).",
        purpose="Financial document Q&A (enhanced confidence)",
        author="ML Engineering",
        variables=["context", "query"],
        model_constraints=["claude-sonnet-4", "gpt-4o"],
        tags=["financial", "rag", "production"],
    ))
    print(f"  Updated: {t1v2.template_id} v{t1v2.version} (hash: {t1v2.content_hash})")

    # Show version history
    print("\n▸ Version history:")
    for v in registry.list_versions("financial_qa"):
        print(f"  v{v['version']} (hash: {v['hash']}) by {v['author']} [{v['status']}]")

    # Show change log
    print("\n▸ Change log:")
    for c in registry.change_log("financial_qa"):
        print(f"  {c['change_type']}: v{c['from_version']} → v{c['to_version']} | {c['diff_summary']}")

    # ── Simulate prompt executions ──
    print("\n▸ Simulating prompt executions...")

    queries = [
        ("What was ACME Corp's revenue in FY2024?",
         "ACME Corp 10-K FY2024: Total revenue was $5.2 billion, up 12% YoY.",
         "Based on the ACME Corp 10-K, total revenue was $5.2 billion in FY2024.",
         "pass", "3.0"),
        ("What was the operating margin?",
         "ACME Corp 10-K FY2024: Operating income was $780M on revenue of $5.2B.",
         "The operating margin was 15% ($780M / $5.2B) based on the 10-K filing.",
         "pass", "3.0"),
        ("Should I invest in ACME?",
         "ACME Corp 10-K FY2024: Total revenue was $5.2 billion.",
         "I cannot provide investment advice. Based on the filing, ACME reported $5.2B revenue.",
         "fail", "3.0"),  # Critic catches near-advice
        ("What was ACME Corp's revenue in FY2024?",
         "ACME Corp 10-K FY2024: Total revenue was $5.2 billion, up 12% YoY.",
         "Based on the ACME Corp 10-K, total revenue was $5.2 billion in FY2024. Confidence: high.",
         "pass", "3.1"),
        ("Summarize ACME's risk factors.",
         "ACME Corp 10-K FY2024: Risk factors include market volatility, regulatory changes...",
         "Key risk factors from ACME's 10-K include: 1) Market volatility, 2) Regulatory changes, 3) Technology disruption. Confidence: high.",
         "pass", "3.1"),
    ]

    for i, (query, context, response, verdict, version) in enumerate(queries):
        ctx = plogger.start_execution(
            template_id="financial_qa",
            variables={"query": query, "context": context},
            session_id=f"session-demo-{i//3 + 1}",
            lineage_id=f"trace-{i+1:04d}",
            request_id=f"req-{i+1:04d}",
            data_tier="internal",
            pii_redactions=1 if i == 2 else 0,
            context_documents=["acme-10k-fy2024"],
        )

        # Force specific template version for demo
        ctx.template_version = version

        plogger.complete_execution(
            ctx.execution_id,
            response_text=response,
            model_id="anthropic/claude-sonnet-4",
            provider="anthropic",
            prompt_tokens=1800 + i * 100,
            completion_tokens=250 + i * 50,
            latency_ms=1900 + i * 200,
            temperature=0.1,
            critic_verdict=verdict,
            critic_violations=["near_investment_advice"] if verdict == "fail" else [],
        )

        print(f"  Execution {i+1}: {ctx.execution_id} | v{version} | "
              f"critic={verdict} | {ctx.total_tokens} tokens")

    # ── Auditor queries ──
    print("\n" + "=" * 70)
    print("AUDITOR QUERIES")
    print("=" * 70)

    # Query 1: Get exact prompt for a specific execution
    print("\n▸ Query 1: 'Show me the exact prompt for the first execution'")
    first_exec = list(plogger._executions.values())[0]
    exe = plogger.get_execution(first_exec.execution_id)
    print(f"  Execution ID: {exe['execution_id']}")
    print(f"  Template: {exe['template_id']} v{exe['template_version']}")
    print(f"  System prompt: {exe['system_prompt'][:80]}...")
    print(f"  User prompt: {exe['user_prompt'][:80]}...")
    print(f"  Response: {exe['response_text'][:80]}...")
    print(f"  Model: {exe['model_id']}")
    print(f"  Critic: {exe['critic_verdict']}")
    print(f"  Cost: ${exe['estimated_cost']:.6f}")

    # Query 2: Get all executions for a session
    print("\n▸ Query 2: 'All prompts in session-demo-1'")
    session_exes = plogger.get_by_session("session-demo-1")
    print(f"  Found: {len(session_exes)} executions")
    for e in session_exes:
        print(f"    [{e['execution_id']}] {e['variables_used'].get('query', '')[:50]}... → {e['critic_verdict']}")

    # Query 3: Get executions linked to a lineage trace
    print("\n▸ Query 3: 'Prompt for lineage trace trace-0003'")
    lineage_exes = plogger.get_by_lineage("trace-0003")
    for e in lineage_exes:
        print(f"  Full prompt sent to LLM:")
        print(f"    {e['full_prompt'][:120]}...")
        print(f"  Critic violations: {e['critic_violations']}")

    # ── Analytics ──
    print("\n" + "=" * 70)
    print("ANALYTICS")
    print("=" * 70)

    # Template performance
    print("\n▸ Template performance (financial_qa):")
    perf = plogger.template_performance("financial_qa")
    print(f"  Total executions: {perf['total_executions']}")
    print(f"  Total cost: ${perf['total_cost']}")
    for ver, stats in perf["versions"].items():
        print(f"  Version {ver}: {stats['executions']} executions, "
              f"pass rate={stats['critic_pass_rate']:.0%}, "
              f"avg latency={stats['avg_latency_ms']}ms, "
              f"avg cost=${stats['avg_cost']:.6f}")

    # A/B comparison
    print("\n▸ A/B Comparison: v3.0 vs v3.1")
    comparison = plogger.compare_versions("financial_qa", "3.0", "3.1")
    for metric, vals in comparison["comparison"].items():
        print(f"  {metric}: v3.0={vals['a']} → v3.1={vals['b']} ({vals.get('delta', '')})")
    print(f"  Recommendation: {comparison['recommendation']}")

    # Audit report
    print("\n▸ Audit report:")
    report = plogger.audit_report()
    print(f"  Total executions: {report['total_executions']}")
    print(f"  By model: {report['by_model']}")
    print(f"  By template: {report['by_template']}")
    print(f"  By data tier: {report['by_data_tier']}")
    print(f"  Total cost: ${report['total_cost']}")
    print(f"  Total tokens: {report['total_tokens']:,}")
    print(f"  PII redactions: {report['pii_redactions_total']}")
    print(f"  Critic verdicts: {report['critic_verdicts']}")
    print(f"  Unique sessions: {report['unique_sessions']}")

    print(f"\n▸ Audit events: {len(audit_log)}")

    print("\n" + "=" * 70)
    print("✓ Prompt logging & auditability demo complete")
    print("=" * 70)


if __name__ == "__main__":
    demo()

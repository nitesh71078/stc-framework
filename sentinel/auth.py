"""
STC Framework - Authentication & Authorization Module

Provides two layers of access control:

1. MACHINE-TO-MACHINE (Persona Auth)
   - Each STC persona (Stalwart, Trainer, Critic) has a scoped identity
   - Permissions are enforced via Casbin RBAC before every action
   - Gateway-level enforcement via virtual keys (LiteLLM/Kong/Bifrost)
   - MCP tool access validated per-persona

2. HUMAN OPERATOR (Operator Auth)
   - RBAC for humans who manage the STC system
   - Roles: admin, operator, auditor, viewer
   - Controls: who can modify the spec, reset suspensions, approve
     routing changes, export audit data
   - Audit trail for every operator action

Built on:
  - Casbin (Apache 2.0) for policy enforcement
  - JWT validation for operator identity (pluggable IdP)
  - Declarative Specification as the policy source of truth

Usage:
    from sentinel.auth import STCAuthManager
    auth = STCAuthManager(spec)

    # Machine: check if Stalwart can invoke a tool
    auth.authorize_persona("stalwart", "tool:invoke", "document_retriever")

    # Human: check if operator can reset suspension
    auth.authorize_operator("jane@company.com", "escalation:reset")
"""

import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

from spec.loader import STCSpec

logger = logging.getLogger("stc.auth")


# ============================================================================
# Casbin Policy Model for STC
# ============================================================================

# Casbin model definition (RBAC with resource roles)
# This gets written to a temp file for Casbin to load
STC_CASBIN_MODEL = """
[request_definition]
r = sub, act, obj

[policy_definition]
p = sub, act, obj

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && r.act == p.act && (r.obj == p.obj || p.obj == "*")
"""


def _generate_casbin_policy(spec: STCSpec) -> str:
    """
    Generate Casbin policy CSV from the Declarative Specification.
    
    This is the bridge between STC's YAML-based policy and Casbin's
    enforcement engine. Every permission in the spec becomes a Casbin rule.
    """
    lines = []

    # ── Persona Permissions (from spec) ───────────────────────────────
    for persona in ["stalwart", "trainer", "critic"]:
        section = spec.raw.get(persona, {})
        permissions = section.get("auth", {}).get("permissions", [])
        for perm in permissions:
            # Parse "llm:call" → act="llm", obj="call"
            # Or keep as single action with wildcard obj
            if ":" in perm:
                act, obj = perm.split(":", 1)
            else:
                act, obj = perm, "*"
            lines.append(f"p, {persona}, {act}, {obj}")

    # ── Tool Access (from spec permitted_tools) ───────────────────────
    for tool in spec.stalwart.get("permitted_tools", []):
        tool_name = tool.get("name", "")
        lines.append(f"p, stalwart, tool, {tool_name}")

    # ── MCP Access Policy ─────────────────────────────────────────────
    for policy in spec.sentinel.get("mcp_access_policy", []):
        tool_name = policy.get("tool", "")
        for persona in policy.get("allowed_personas", []):
            lines.append(f"p, {persona}, mcp, {tool_name}")

    # ── Human Operator Roles ──────────────────────────────────────────
    # admin: full access
    lines.append("p, admin, spec, modify")
    lines.append("p, admin, spec, read")
    lines.append("p, admin, escalation, reset")
    lines.append("p, admin, routing, approve")
    lines.append("p, admin, routing, modify")
    lines.append("p, admin, audit, export")
    lines.append("p, admin, audit, read")
    lines.append("p, admin, keys, rotate")
    lines.append("p, admin, keys, revoke")
    lines.append("p, admin, keys, create")
    lines.append("p, admin, system, suspend")
    lines.append("p, admin, system, resume")

    # operator: can manage day-to-day operations
    lines.append("p, operator, spec, read")
    lines.append("p, operator, escalation, reset")
    lines.append("p, operator, routing, approve")
    lines.append("p, operator, audit, read")
    lines.append("p, operator, keys, rotate")
    lines.append("p, operator, system, resume")

    # auditor: read-only access to everything for compliance review
    lines.append("p, auditor, spec, read")
    lines.append("p, auditor, audit, export")
    lines.append("p, auditor, audit, read")
    lines.append("p, auditor, traces, read")
    lines.append("p, auditor, guardrails, read")

    # viewer: minimal read access
    lines.append("p, viewer, spec, read")
    lines.append("p, viewer, audit, read")

    return "\n".join(lines)


# ============================================================================
# API Key Management
# ============================================================================

@dataclass
class APIKey:
    """Represents an STC API key for a persona or operator."""
    key_id: str
    key_hash: str  # SHA-256 hash — never store plaintext
    persona: str  # stalwart | trainer | critic | operator:<email>
    scope: str  # From spec auth.key_scope
    created_at: str
    expires_at: str
    active: bool = True
    last_used: Optional[str] = None
    usage_count: int = 0


class KeyManager:
    """
    Manages API keys for STC personas and human operators.
    
    Keys are:
      - Generated with cryptographic randomness
      - Stored as SHA-256 hashes (never plaintext after creation)
      - Scoped to a persona with defined permissions
      - Rotatable per spec (sentinel.auth.key_rotation_days)
      - Revocable with audit trail
    """

    def __init__(self, spec: STCSpec, storage_path: str = ".stc_keys"):
        self.spec = spec
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.keys: dict[str, APIKey] = {}
        self._load_keys()

    def generate_key(self, persona: str, scope: str = "",
                     expiry_days: Optional[int] = None) -> tuple[str, APIKey]:
        """
        Generate a new API key for a persona.
        Returns (plaintext_key, key_record).
        
        IMPORTANT: The plaintext key is returned ONCE. After this,
        only the hash is stored.
        """
        rotation_days = self.spec.sentinel.get("auth", {}).get("key_rotation_days", 90)
        if expiry_days is None:
            expiry_days = rotation_days

        # Generate cryptographically secure key
        plaintext = f"sk-stc-{persona}-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        key_id = f"key-{persona}-{secrets.token_hex(8)}"

        now = datetime.now(timezone.utc)
        key_record = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            persona=persona,
            scope=scope or f"{persona}-scope",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=expiry_days)).isoformat(),
        )

        self.keys[key_id] = key_record
        self._save_keys()

        logger.info(f"Generated API key {key_id} for persona '{persona}' (expires in {expiry_days}d)")
        return plaintext, key_record

    def validate_key(self, plaintext_key: str) -> Optional[APIKey]:
        """
        Validate an API key. Returns the key record if valid, None otherwise.
        """
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

        for key_record in self.keys.values():
            if key_record.key_hash == key_hash:
                # Check active
                if not key_record.active:
                    logger.warning(f"Key {key_record.key_id} is revoked")
                    return None

                # Check expiry
                expires = datetime.fromisoformat(key_record.expires_at)
                if datetime.now(timezone.utc) > expires:
                    logger.warning(f"Key {key_record.key_id} has expired")
                    return None

                # Update usage
                key_record.last_used = datetime.now(timezone.utc).isoformat()
                key_record.usage_count += 1
                self._save_keys()

                return key_record

        return None

    def revoke_key(self, key_id: str, revoked_by: str) -> bool:
        """Revoke an API key."""
        if key_id in self.keys:
            self.keys[key_id].active = False
            self._save_keys()
            logger.info(f"Key {key_id} revoked by {revoked_by}")
            return True
        return False

    def rotate_keys(self, persona: str) -> tuple[str, APIKey]:
        """Revoke all existing keys for a persona and generate a new one."""
        for key_id, key_record in self.keys.items():
            if key_record.persona == persona and key_record.active:
                key_record.active = False

        new_key, new_record = self.generate_key(persona)
        self._save_keys()
        logger.info(f"Rotated keys for persona '{persona}'")
        return new_key, new_record

    def get_expiring_keys(self, within_days: int = 14) -> list[APIKey]:
        """Get keys that will expire within the specified number of days."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=within_days)
        return [
            k for k in self.keys.values()
            if k.active and datetime.fromisoformat(k.expires_at) < cutoff
        ]

    def _load_keys(self):
        keys_file = self.storage_path / "keys.json"
        if keys_file.exists():
            with open(keys_file, "r") as f:
                data = json.load(f)
            self.keys = {
                k: APIKey(**v) for k, v in data.items()
            }

    def _save_keys(self):
        keys_file = self.storage_path / "keys.json"
        with open(keys_file, "w") as f:
            json.dump(
                {k: vars(v) for k, v in self.keys.items()},
                f, indent=2,
            )


# ============================================================================
# Operator Identity (JWT Validation)
# ============================================================================

@dataclass
class OperatorIdentity:
    """Represents an authenticated human operator."""
    email: str
    name: str
    roles: list[str]  # admin | operator | auditor | viewer
    authenticated_at: str
    auth_method: str  # jwt | api_key | local


class OperatorAuthenticator:
    """
    Authenticates human operators. Supports:
    - JWT validation (for IdP integration: Okta, Azure AD, Auth0)
    - API key (for CLI/automation access)
    - Local auth (for development/testing)
    """

    def __init__(self, spec: STCSpec):
        self.spec = spec
        self.operator_config = spec.sentinel.get("auth", {}).get("operators", {})

        # Load local operator definitions (for dev/testing)
        self.local_operators = self.operator_config.get("local_operators", {})

    def authenticate_jwt(self, token: str) -> Optional[OperatorIdentity]:
        """
        Validate a JWT token from an external IdP.
        
        In production, this validates against:
        - Okta: OIDC discovery + JWKS
        - Azure AD: Microsoft identity platform
        - Auth0: Tenant JWKS endpoint
        - KeyCloak: Realm JWKS endpoint
        """
        try:
            import jwt as pyjwt

            idp_config = self.operator_config.get("idp", {})
            jwks_uri = idp_config.get("jwks_uri", "")
            issuer = idp_config.get("issuer", "")
            audience = idp_config.get("audience", "stc-framework")

            if not jwks_uri:
                logger.warning("No IdP configured; JWT validation unavailable")
                return None

            # In production: fetch JWKS, validate signature, check claims
            # For now, decode without verification as a stub
            claims = pyjwt.decode(token, options={"verify_signature": False})

            email = claims.get("email", claims.get("sub", ""))
            name = claims.get("name", email)

            # Map IdP roles/groups to STC roles
            idp_roles = claims.get("roles", claims.get("groups", []))
            stc_roles = self._map_idp_roles(idp_roles)

            return OperatorIdentity(
                email=email,
                name=name,
                roles=stc_roles,
                authenticated_at=datetime.now(timezone.utc).isoformat(),
                auth_method="jwt",
            )

        except ImportError:
            logger.warning("PyJWT not installed; JWT auth unavailable. pip install PyJWT")
            return None
        except Exception as e:
            logger.error(f"JWT validation failed: {e}")
            return None

    def authenticate_local(self, email: str) -> Optional[OperatorIdentity]:
        """
        Local authentication for development/testing.
        Operators defined in the Declarative Specification.
        """
        if email in self.local_operators:
            operator_def = self.local_operators[email]
            return OperatorIdentity(
                email=email,
                name=operator_def.get("name", email),
                roles=operator_def.get("roles", ["viewer"]),
                authenticated_at=datetime.now(timezone.utc).isoformat(),
                auth_method="local",
            )
        return None

    def _map_idp_roles(self, idp_roles: list[str]) -> list[str]:
        """Map external IdP roles/groups to STC operator roles."""
        role_mapping = self.operator_config.get("role_mapping", {
            "stc-admins": "admin",
            "stc-operators": "operator",
            "stc-auditors": "auditor",
            "stc-viewers": "viewer",
        })

        stc_roles = []
        for idp_role in idp_roles:
            if idp_role in role_mapping:
                stc_roles.append(role_mapping[idp_role])

        return stc_roles or ["viewer"]  # Default to viewer


# ============================================================================
# Audit Trail for Auth Events
# ============================================================================

@dataclass
class AuthEvent:
    """Immutable record of an authorization decision."""
    timestamp: str
    subject: str  # persona name or operator email
    action: str
    resource: str
    decision: str  # allow | deny
    reason: str = ""
    context: dict = field(default_factory=dict)


class AuthAuditLog:
    """Append-only audit log for all authorization decisions."""

    def __init__(self, log_path: str = ".stc_auth_audit"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)
        self.events: list[AuthEvent] = []

    def record(self, event: AuthEvent):
        """Record an authorization event (append-only)."""
        self.events.append(event)

        # Write to daily log file
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_path / f"auth_events_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(vars(event)) + "\n")

    def get_events(self, subject: Optional[str] = None,
                   action: Optional[str] = None,
                   since_hours: int = 24) -> list[AuthEvent]:
        """Query recent auth events."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        filtered = []
        for e in self.events:
            event_time = datetime.fromisoformat(e.timestamp)
            if event_time < cutoff:
                continue
            if subject and e.subject != subject:
                continue
            if action and e.action != action:
                continue
            filtered.append(e)
        return filtered


# ============================================================================
# Main Auth Manager
# ============================================================================

class STCAuthManager:
    """
    Centralized authentication and authorization for the STC Framework.
    
    Combines:
    - Casbin RBAC for policy enforcement
    - API key management for persona identity
    - JWT/local auth for human operators
    - Audit logging for all auth decisions
    
    Every authorization check goes through this manager, producing
    an auditable record regardless of outcome.
    """

    def __init__(self, spec: STCSpec):
        self.spec = spec
        self.key_manager = KeyManager(spec)
        self.operator_auth = OperatorAuthenticator(spec)
        self.audit_log = AuthAuditLog()
        self._setup_casbin(spec)

    def _setup_casbin(self, spec: STCSpec):
        """Initialize Casbin enforcer with STC policies."""
        try:
            import casbin

            # Write model and policy to temp files
            import tempfile
            model_path = Path(tempfile.mkdtemp()) / "stc_model.conf"
            policy_path = model_path.parent / "stc_policy.csv"

            model_path.write_text(STC_CASBIN_MODEL)
            policy_path.write_text(_generate_casbin_policy(spec))

            self.enforcer = casbin.Enforcer(str(model_path), str(policy_path))
            self.casbin_available = True
            logger.info("Casbin RBAC enforcer initialized")

        except ImportError:
            logger.warning(
                "Casbin not installed; authorization will use fallback checks. "
                "Run: pip install casbin"
            )
            self.casbin_available = False
            self.enforcer = None

    # ── Persona Authorization ─────────────────────────────────────────

    def authorize_persona(self, persona: str, action: str,
                          resource: str = "*") -> bool:
        """
        Check if a persona is authorized to perform an action.
        
        Examples:
            authorize_persona("stalwart", "llm", "call")
            authorize_persona("stalwart", "tool", "document_retriever")
            authorize_persona("trainer", "routing", "write")
            authorize_persona("critic", "escalation", "trigger")
        """
        if self.casbin_available:
            allowed = self.enforcer.enforce(persona, action, resource)
        else:
            # Fallback: check against spec permissions
            allowed = self._fallback_persona_check(persona, action, resource)

        # Audit
        self.audit_log.record(AuthEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            subject=persona,
            action=action,
            resource=resource,
            decision="allow" if allowed else "deny",
            context={"auth_type": "persona", "engine": "casbin" if self.casbin_available else "fallback"},
        ))

        if not allowed:
            logger.warning(f"DENIED: persona={persona}, action={action}, resource={resource}")

        return allowed

    def authorize_tool_access(self, persona: str, tool_name: str) -> bool:
        """Check if a persona can access a specific tool/MCP server."""
        return self.authorize_persona(persona, "tool", tool_name)

    def authorize_mcp_access(self, persona: str, mcp_server: str) -> bool:
        """Check if a persona can access a specific MCP server."""
        return self.authorize_persona(persona, "mcp", mcp_server)

    # ── Operator Authorization ────────────────────────────────────────

    def authorize_operator(self, operator_email: str, action: str,
                           resource: str = "*") -> bool:
        """
        Check if a human operator is authorized to perform an action.
        
        Examples:
            authorize_operator("jane@co.com", "spec", "modify")
            authorize_operator("jane@co.com", "escalation", "reset")
            authorize_operator("bob@co.com", "audit", "export")
        """
        # Look up operator's roles
        operator = self.operator_auth.authenticate_local(operator_email)
        if not operator:
            self.audit_log.record(AuthEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                subject=operator_email,
                action=action,
                resource=resource,
                decision="deny",
                reason="operator_not_found",
            ))
            return False

        # Check each role
        allowed = False
        for role in operator.roles:
            if self.casbin_available:
                if self.enforcer.enforce(role, action, resource):
                    allowed = True
                    break
            else:
                if self._fallback_operator_check(role, action, resource):
                    allowed = True
                    break

        self.audit_log.record(AuthEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            subject=operator_email,
            action=action,
            resource=resource,
            decision="allow" if allowed else "deny",
            context={
                "auth_type": "operator",
                "roles": operator.roles,
                "auth_method": operator.auth_method,
            },
        ))

        if not allowed:
            logger.warning(
                f"DENIED: operator={operator_email}, roles={operator.roles}, "
                f"action={action}, resource={resource}"
            )

        return allowed

    # ── Key Operations ────────────────────────────────────────────────

    def generate_persona_keys(self) -> dict[str, str]:
        """Generate API keys for all three personas. Returns persona→plaintext map."""
        keys = {}
        for persona in ["stalwart", "trainer", "critic"]:
            scope = self.spec.raw.get(persona, {}).get("auth", {}).get("key_scope", "")
            plaintext, _ = self.key_manager.generate_key(persona, scope)
            keys[persona] = plaintext
        return keys

    def validate_api_key(self, key: str) -> Optional[str]:
        """Validate an API key and return the persona name, or None."""
        record = self.key_manager.validate_key(key)
        if record:
            return record.persona
        return None

    # ── Convenience: Pre-built Auth Checks ────────────────────────────

    def can_stalwart_call_llm(self) -> bool:
        return self.authorize_persona("stalwart", "llm", "call")

    def can_stalwart_invoke_tool(self, tool_name: str) -> bool:
        return self.authorize_persona("stalwart", "tool", tool_name)

    def can_trainer_update_routing(self) -> bool:
        return self.authorize_persona("trainer", "routing", "write")

    def can_trainer_write_prompts(self) -> bool:
        return self.authorize_persona("trainer", "prompts", "write")

    def can_critic_trigger_escalation(self) -> bool:
        return self.authorize_persona("critic", "escalation", "trigger")

    def can_operator_modify_spec(self, email: str) -> bool:
        return self.authorize_operator(email, "spec", "modify")

    def can_operator_reset_escalation(self, email: str) -> bool:
        return self.authorize_operator(email, "escalation", "reset")

    def can_operator_export_audit(self, email: str) -> bool:
        return self.authorize_operator(email, "audit", "export")

    # ── Fallback Checks (when Casbin not available) ───────────────────

    def _fallback_persona_check(self, persona: str, action: str,
                                 resource: str) -> bool:
        """Check permissions against the spec when Casbin is not installed."""
        section = self.spec.raw.get(persona, {})
        permissions = section.get("auth", {}).get("permissions", [])
        perm_string = f"{action}:{resource}" if resource != "*" else action
        return perm_string in permissions or f"{action}:*" in permissions

    def _fallback_operator_check(self, role: str, action: str,
                                  resource: str) -> bool:
        """Hardcoded role checks when Casbin is not available."""
        role_permissions = {
            "admin": {"spec:modify", "spec:read", "escalation:reset",
                      "routing:approve", "routing:modify", "audit:export",
                      "audit:read", "keys:rotate", "keys:revoke", "keys:create",
                      "system:suspend", "system:resume"},
            "operator": {"spec:read", "escalation:reset", "routing:approve",
                         "audit:read", "keys:rotate", "system:resume"},
            "auditor": {"spec:read", "audit:export", "audit:read",
                        "traces:read", "guardrails:read"},
            "viewer": {"spec:read", "audit:read"},
        }
        perms = role_permissions.get(role, set())
        return f"{action}:{resource}" in perms or f"{action}:*" in perms

    # ── Health / Status ───────────────────────────────────────────────

    def get_auth_status(self) -> dict:
        """Get current auth system status."""
        expiring = self.key_manager.get_expiring_keys(within_days=14)
        recent_denials = self.audit_log.get_events(since_hours=24)
        denial_count = sum(1 for e in recent_denials if e.decision == "deny")

        return {
            "casbin_available": self.casbin_available,
            "total_keys": len(self.key_manager.keys),
            "active_keys": sum(1 for k in self.key_manager.keys.values() if k.active),
            "expiring_soon": len(expiring),
            "denials_24h": denial_count,
            "total_events_24h": len(recent_denials),
        }


# ============================================================================
# Declarative Specification Extension for Auth
# ============================================================================

SPEC_AUTH_EXAMPLE = """
# Add to your stc-spec.yaml under the sentinel section:

sentinel:
  auth:
    virtual_keys: true
    key_rotation_days: 90
    
    persona_keys:
      stalwart: "${STC_STALWART_KEY}"
      trainer: "${STC_TRAINER_KEY}"
      critic: "${STC_CRITIC_KEY}"
    
    # Human operator configuration
    operators:
      # Identity Provider (for production)
      idp:
        provider: okta  # okta | azure_ad | auth0 | keycloak
        jwks_uri: "https://your-org.okta.com/oauth2/default/v1/keys"
        issuer: "https://your-org.okta.com/oauth2/default"
        audience: "stc-framework"
      
      # Map IdP groups/roles to STC operator roles
      role_mapping:
        "stc-admins": admin
        "stc-operators": operator
        "stc-auditors": auditor
        "stc-viewers": viewer
      
      # Local operators (for development/testing only)
      local_operators:
        "admin@company.com":
          name: "Admin User"
          roles: [admin]
        "operator@company.com":
          name: "STC Operator"
          roles: [operator]
        "auditor@company.com":
          name: "Compliance Auditor"
          roles: [auditor]
        "viewer@company.com":
          name: "Read-Only User"
          roles: [viewer]
"""


# ============================================================================
# Entry Point / Demo
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from spec.loader import load_spec

    spec = load_spec("spec/stc-spec.yaml")
    auth = STCAuthManager(spec)

    print("=" * 60)
    print("  STC Auth Manager — Policy Enforcement Demo")
    print("=" * 60)

    # Persona checks
    print("\n  PERSONA AUTHORIZATION:")
    checks = [
        ("stalwart", "llm", "call"),
        ("stalwart", "tool", "document_retriever"),
        ("stalwart", "tool", "unauthorized_tool"),
        ("stalwart", "routing", "write"),  # Should deny
        ("trainer", "traces", "read"),
        ("trainer", "routing", "write"),
        ("trainer", "escalation", "trigger"),  # Should deny
        ("critic", "guardrails", "invoke"),
        ("critic", "escalation", "trigger"),
        ("critic", "routing", "write"),  # Should deny
    ]

    for persona, action, resource in checks:
        allowed = auth.authorize_persona(persona, action, resource)
        icon = "✅" if allowed else "🚫"
        print(f"    {icon} {persona:10s} → {action}:{resource}")

    # Operator checks
    print("\n  OPERATOR AUTHORIZATION:")
    # Add test operators to spec for demo
    spec.raw.setdefault("sentinel", {}).setdefault("auth", {}).setdefault("operators", {})
    spec.raw["sentinel"]["auth"]["operators"]["local_operators"] = {
        "admin@co.com": {"name": "Admin", "roles": ["admin"]},
        "ops@co.com": {"name": "Operator", "roles": ["operator"]},
        "audit@co.com": {"name": "Auditor", "roles": ["auditor"]},
        "view@co.com": {"name": "Viewer", "roles": ["viewer"]},
    }
    auth.operator_auth = OperatorAuthenticator(spec)

    operator_checks = [
        ("admin@co.com", "spec", "modify"),
        ("admin@co.com", "escalation", "reset"),
        ("ops@co.com", "escalation", "reset"),
        ("ops@co.com", "spec", "modify"),  # Should deny
        ("audit@co.com", "audit", "export"),
        ("audit@co.com", "spec", "modify"),  # Should deny
        ("view@co.com", "audit", "read"),
        ("view@co.com", "escalation", "reset"),  # Should deny
    ]

    for email, action, resource in operator_checks:
        allowed = auth.authorize_operator(email, action, resource)
        icon = "✅" if allowed else "🚫"
        print(f"    {icon} {email:18s} → {action}:{resource}")

    # Auth status
    print(f"\n  AUTH STATUS:")
    status = auth.get_auth_status()
    for k, v in status.items():
        print(f"    {k}: {v}")

    # Recent events
    events = auth.audit_log.get_events(since_hours=1)
    print(f"\n  AUDIT: {len(events)} auth events recorded in the last hour")
    denials = [e for e in events if e.decision == "deny"]
    print(f"         {len(denials)} denials")

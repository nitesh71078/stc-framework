"""
STC Framework — Secret Management Module
sentinel/secret_manager.py

Pluggable secret backends: HashiCorp Vault | AWS Secrets Manager | Env (dev).
Supports automatic rotation, encrypted token maps, and audit trail integration.
Part of the Sentinel Layer (infrastructure enforcement, not intelligence).
"""

import os, json, time, hashlib, logging
import secrets as stdlib_secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("stc.sentinel.secrets")


class SecretType(Enum):
    LLM_PROVIDER_KEY = "llm_provider_key"
    PERSONA_API_KEY = "persona_api_key"
    DATABASE_CREDENTIAL = "database_credential"
    TLS_CERTIFICATE = "tls_certificate"
    ENCRYPTION_KEY = "encryption_key"
    IDP_CLIENT_SECRET = "idp_client_secret"
    WEBHOOK_TOKEN = "webhook_token"


@dataclass
class SecretMetadata:
    secret_id: str
    secret_type: SecretType
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_rotated: Optional[datetime] = None
    rotation_days: int = 90
    version: int = 1
    fingerprint: str = ""

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and datetime.now(timezone.utc) >= self.expires_at

    @property
    def days_until_expiry(self) -> Optional[int]:
        if self.expires_at is None:
            return None
        return max(0, (self.expires_at - datetime.now(timezone.utc)).days)

    @property
    def needs_rotation(self) -> bool:
        if self.last_rotated is None:
            return False
        threshold = timedelta(days=int(self.rotation_days * 0.8))
        return datetime.now(timezone.utc) - self.last_rotated >= threshold


class SecretBackend(ABC):
    @abstractmethod
    def get_secret(self, path: str) -> Optional[str]: ...
    @abstractmethod
    def set_secret(self, path: str, value: str, metadata: Optional[Dict] = None) -> bool: ...
    @abstractmethod
    def delete_secret(self, path: str) -> bool: ...
    @abstractmethod
    def list_secrets(self, prefix: str = "") -> List[str]: ...
    @abstractmethod
    def health_check(self) -> Dict[str, Any]: ...


class VaultBackend(SecretBackend):
    """HashiCorp Vault KV v2 backend."""
    def __init__(self, vault_addr: str, mount_path: str = "secret",
                 base_path: str = "stc", token: Optional[str] = None,
                 role_id: Optional[str] = None, secret_id: Optional[str] = None):
        self.vault_addr = vault_addr.rstrip("/")
        self.mount_path = mount_path
        self.base_path = base_path
        self._token = token
        if not self._token and role_id and secret_id:
            import requests
            resp = requests.post(f"{self.vault_addr}/v1/auth/approle/login",
                                 json={"role_id": role_id, "secret_id": secret_id}, timeout=10)
            resp.raise_for_status()
            self._token = resp.json()["auth"]["client_token"]
        if not self._token:
            raise ValueError("Vault requires VAULT_TOKEN or VAULT_ROLE_ID + VAULT_SECRET_ID")

    def _h(self): return {"X-Vault-Token": self._token, "Content-Type": "application/json"}
    def _url(self, path): return f"{self.vault_addr}/v1/{self.mount_path}/data/{self.base_path}/{path}"

    def get_secret(self, path):
        try:
            import requests
            r = requests.get(self._url(path), headers=self._h(), timeout=10)
            return r.json().get("data", {}).get("data", {}).get("value") if r.status_code != 404 else None
        except Exception as e:
            logger.error(f"Vault get failed: {e}"); return None

    def set_secret(self, path, value, metadata=None):
        try:
            import requests
            payload = {"data": {"value": value, **({"metadata": json.dumps(metadata)} if metadata else {})}}
            requests.post(self._url(path), headers=self._h(), json=payload, timeout=10).raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Vault set failed: {e}"); return False

    def delete_secret(self, path):
        try:
            import requests
            requests.delete(f"{self.vault_addr}/v1/{self.mount_path}/metadata/{self.base_path}/{path}",
                            headers=self._h(), timeout=10).raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Vault delete failed: {e}"); return False

    def list_secrets(self, prefix=""):
        try:
            import requests
            r = requests.request("LIST", f"{self.vault_addr}/v1/{self.mount_path}/metadata/{self.base_path}/{prefix}",
                                 headers=self._h(), timeout=10)
            return r.json().get("data", {}).get("keys", []) if r.status_code != 404 else []
        except Exception as e:
            logger.error(f"Vault list failed: {e}"); return []

    def health_check(self):
        try:
            import requests
            d = requests.get(f"{self.vault_addr}/v1/sys/health", timeout=5).json()
            return {"backend": "vault", "status": "healthy" if not d.get("sealed") else "sealed"}
        except Exception as e:
            return {"backend": "vault", "status": "unreachable", "error": str(e)}


class AWSSecretsBackend(SecretBackend):
    """AWS Secrets Manager backend."""
    def __init__(self, region="us-east-1", prefix="stc"):
        self.prefix, self.region, self._client = prefix, region, None

    def _c(self):
        if not self._client:
            import boto3; self._client = boto3.client("secretsmanager", region_name=self.region)
        return self._client

    def get_secret(self, path):
        try: return self._c().get_secret_value(SecretId=f"{self.prefix}/{path}").get("SecretString")
        except: return None

    def set_secret(self, path, value, metadata=None):
        try:
            name = f"{self.prefix}/{path}"
            try: self._c().put_secret_value(SecretId=name, SecretString=value)
            except: self._c().create_secret(Name=name, SecretString=value)
            return True
        except: return False

    def delete_secret(self, path):
        try: self._c().delete_secret(SecretId=f"{self.prefix}/{path}", ForceDeleteWithoutRecovery=True); return True
        except: return False

    def list_secrets(self, prefix=""):
        try:
            keys = []
            for page in self._c().get_paginator("list_secrets").paginate(
                    Filters=[{"Key": "name", "Values": [f"{self.prefix}/{prefix}"]}]):
                keys.extend(s["Name"][len(self.prefix)+1:] for s in page.get("SecretList", [])
                            if s["Name"].startswith(self.prefix + "/"))
            return keys
        except: return []

    def health_check(self):
        try: self._c().list_secrets(MaxResults=1); return {"backend": "aws", "status": "healthy"}
        except Exception as e: return {"backend": "aws", "status": "unreachable", "error": str(e)}


class EnvBackend(SecretBackend):
    """Environment variable backend (development)."""
    PREFIX = "STC_SECRET_"
    def _n(self, p): return self.PREFIX + p.upper().replace("/", "_").replace("-", "_")
    def get_secret(self, path): return os.environ.get(self._n(path))
    def set_secret(self, path, value, metadata=None): os.environ[self._n(path)] = value; return True
    def delete_secret(self, path):
        n = self._n(path)
        if n in os.environ: del os.environ[n]; return True
        return False
    def list_secrets(self, prefix=""):
        fp = self._n(prefix)
        return [k[len(self.PREFIX):].lower().replace("_", "/") for k in os.environ if k.startswith(fp)]
    def health_check(self):
        return {"backend": "env", "status": "healthy",
                "secrets_found": sum(1 for k in os.environ if k.startswith(self.PREFIX))}


class SecretManager:
    """Central secret management for all STC components."""

    def __init__(self, backend: SecretBackend, audit_callback=None):
        self.backend = backend
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        self._cache_ttl = 300
        self._audit_callback = audit_callback

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], audit_callback=None) -> "SecretManager":
        cfg = spec.get("sentinel", {}).get("secrets", {})
        engine = cfg.get("engine", "env")
        if engine == "vault":
            backend = VaultBackend(
                vault_addr=cfg.get("vault_addr", os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")),
                mount_path=cfg.get("vault_mount", "secret"), base_path=cfg.get("vault_base_path", "stc"),
                token=os.environ.get("VAULT_TOKEN"), role_id=os.environ.get("VAULT_ROLE_ID"),
                secret_id=os.environ.get("VAULT_SECRET_ID"))
        elif engine == "aws_secrets_manager":
            backend = AWSSecretsBackend(region=cfg.get("aws_region", "us-east-1"), prefix=cfg.get("aws_prefix", "stc"))
        else:
            backend = EnvBackend()
        return cls(backend=backend, audit_callback=audit_callback)

    def get(self, path: str, bypass_cache: bool = False) -> Optional[str]:
        if not bypass_cache and path in self._cache:
            val, t = self._cache[path]
            if time.time() - t < self._cache_ttl: return val
        val = self.backend.get_secret(path)
        if val is not None:
            self._cache[path] = (val, time.time())
            self._emit("secret_accessed", path)
        return val

    def set(self, path: str, value: str, secret_type: SecretType, rotation_days: int = 90) -> bool:
        now = datetime.now(timezone.utc)
        fp = hashlib.sha256(value.encode()).hexdigest()[:16]
        ver = (self._metadata[path].version + 1) if path in self._metadata else 1
        meta = SecretMetadata(secret_id=path, secret_type=secret_type, created_at=now,
                              expires_at=now + timedelta(days=rotation_days), last_rotated=now,
                              rotation_days=rotation_days, version=ver, fingerprint=fp)
        ok = self.backend.set_secret(path, value, {"type": secret_type.value, "version": str(ver)})
        if ok:
            self._metadata[path] = meta
            self._cache[path] = (value, time.time())
            self._emit("secret_stored", path, {"type": secret_type.value, "version": ver, "fingerprint": fp})
        return ok

    def rotate(self, path: str) -> Optional[str]:
        meta = self._metadata.get(path)
        if not meta: return None
        new_val = stdlib_secrets.token_urlsafe(48)
        old_ver = meta.version
        if self.set(path, new_val, meta.secret_type, meta.rotation_days):
            self._emit("secret_rotated", path, {"old_version": old_ver, "new_version": old_ver + 1})
            return new_val
        return None

    def rotate_if_needed(self, path: str) -> Optional[str]:
        meta = self._metadata.get(path)
        return self.rotate(path) if meta and meta.needs_rotation else None

    def check_all_rotations(self) -> List[Dict[str, Any]]:
        events = []
        for path, meta in list(self._metadata.items()):
            if meta.is_expired:
                events.append({"path": path, "status": "EXPIRED", "days_until_expiry": 0})
                self.rotate(path)
            elif meta.needs_rotation:
                events.append({"path": path, "status": "ROTATION_DUE", "days_until_expiry": meta.days_until_expiry})
                self.rotate(path)
            else:
                events.append({"path": path, "status": "HEALTHY", "days_until_expiry": meta.days_until_expiry})
        return events

    def health_check(self) -> Dict[str, Any]:
        bh = self.backend.health_check()
        expired = [p for p, m in self._metadata.items() if m.is_expired]
        expiring = [p for p, m in self._metadata.items()
                    if m.days_until_expiry is not None and m.days_until_expiry < 7 and not m.is_expired]
        return {"backend": bh, "total_secrets": len(self._metadata), "expired": expired,
                "expiring_soon": expiring, "cache_size": len(self._cache),
                "status": "critical" if expired else ("warning" if expiring else "healthy")}

    def _emit(self, event_type, path, details=None):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "component": "sentinel.secrets",
                 "event_type": event_type, "secret_path": path, "details": details or {}}
        if self._audit_callback: self._audit_callback(event)


class EncryptedTokenStore:
    """Encrypted PII surrogate token map. Session-scoped, destroyed at session end."""

    def __init__(self, secret_manager: SecretManager, key_path: str = "encryption/token_map"):
        self.secret_manager, self.key_path = secret_manager, key_path
        self._data: Dict[str, str] = {}
        self._use_aes = False
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
            self._use_aes = True
        except ImportError:
            logger.warning("cryptography not installed — XOR obfuscation only (dev)")

    def store(self, original: str, surrogate: str): self._data[surrogate] = original
    def retrieve(self, surrogate: str) -> Optional[str]: return self._data.get(surrogate)
    def __len__(self): return len(self._data)

    def export_encrypted(self) -> bytes:
        pt = json.dumps(self._data).encode()
        if self._use_aes:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            kh = self.secret_manager.get(self.key_path)
            if not kh:
                k = AESGCM.generate_key(bit_length=256)
                self.secret_manager.set(self.key_path, k.hex(), SecretType.ENCRYPTION_KEY, 365)
                kh = k.hex()
            nonce = os.urandom(12)
            return nonce + AESGCM(bytes.fromhex(kh)).encrypt(nonce, pt, None)
        key = hashlib.sha256(b"stc-dev-key").digest()
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(pt))

    def import_encrypted(self, blob: bytes):
        if self._use_aes:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            kh = self.secret_manager.get(self.key_path)
            if not kh: raise ValueError("Encryption key not found")
            self._data = json.loads(AESGCM(bytes.fromhex(kh)).decrypt(blob[:12], blob[12:], None))
        else:
            key = hashlib.sha256(b"stc-dev-key").digest()
            self._data = json.loads(bytes(b ^ key[i % len(key)] for i, b in enumerate(blob)))

    def destroy(self):
        for k in self._data: self._data[k] = "x" * len(self._data[k])
        self._data.clear()


# ── Demo ────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 70)
    print("STC Secret Management — Demo")
    print("=" * 70)

    audit_log = []
    manager = SecretManager.from_spec(
        {"sentinel": {"secrets": {"engine": "env"}}},
        audit_callback=lambda e: audit_log.append(e))

    print("\n▸ Storing LLM provider keys...")
    manager.set("llm_provider/openai", "sk-test-openai-key-12345", SecretType.LLM_PROVIDER_KEY, 90)
    manager.set("llm_provider/anthropic", "sk-ant-test-key-67890", SecretType.LLM_PROVIDER_KEY, 90)

    print("▸ Storing persona API keys...")
    for p in ["stalwart", "trainer", "critic"]:
        manager.set(f"persona/{p}", stdlib_secrets.token_urlsafe(32), SecretType.PERSONA_API_KEY, 30)

    print("\n▸ Retrieving secrets...")
    k = manager.get("llm_provider/openai")
    print(f"  OpenAI key: {k[:10]}...{k[-5:]}")

    print("\n▸ Health check:")
    h = manager.health_check()
    print(f"  Status: {h['status']}  |  Backend: {h['backend']['status']}  |  Total: {h['total_secrets']}")

    print("\n▸ Checking rotations...")
    for r in manager.check_all_rotations():
        print(f"  {r['path']}: {r['status']} (expires in {r.get('days_until_expiry', 'N/A')} days)")

    print("\n▸ Force rotating persona/stalwart...")
    nk = manager.rotate("persona/stalwart")
    print(f"  New key: {nk[:10]}...{nk[-5:]}")

    print("\n▸ Encrypted token map...")
    ts = EncryptedTokenStore(manager)
    ts.store("John Smith", "SUR_PERSON_a1b2c3")
    ts.store("555-0123", "SUR_PHONE_d4e5f6")
    blob = ts.export_encrypted()
    print(f"  {len(ts)} mappings → {len(blob)} bytes encrypted")
    ts2 = EncryptedTokenStore(manager)
    ts2.import_encrypted(blob)
    assert ts2.retrieve("SUR_PERSON_a1b2c3") == "John Smith"
    print("  ✓ Verified round-trip")
    ts.destroy()
    print("  ✓ Token map destroyed")

    print(f"\n▸ Audit trail: {len(audit_log)} events")
    for e in audit_log[:5]:
        print(f"  [{e['event_type']}] {e['secret_path']}")
    if len(audit_log) > 5:
        print(f"  ... +{len(audit_log) - 5} more")

    print("\n" + "=" * 70)
    print("✓ Secret management demo complete")
    print("=" * 70)


if __name__ == "__main__":
    demo()

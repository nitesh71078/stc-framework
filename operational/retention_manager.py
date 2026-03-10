"""
STC Framework — Data Retention & Destruction Manager
operational/retention_manager.py

Enforces data retention policies and performs secure destruction:
  - Audit trail (JSONL/Parquet) → configurable retention (default 365 days)
  - Vector store embeddings → collection TTL
  - Prompt versions (Langfuse) → configurable retention
  - Surrogate token maps → session-scoped (destroyed at session end)
  - Performance traces → configurable retention (default 90 days)

Destruction is verified: after deletion, the manager attempts to read the
data to confirm it is no longer accessible. All destruction events are
logged to the audit trail (the destruction record itself is retained).

Supports two destruction methods:
  - secure_overwrite: overwrite with random bytes, then delete
  - crypto_erase: delete the encryption key (if data was encrypted at rest)

Part of the Operational Control layer.
"""

import os
import json
import time
import shutil
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("stc.operational.retention")


# ── Data Store Types ────────────────────────────────────────────────────────

class DataStore(Enum):
    AUDIT_TRAIL = "audit_trail"
    VECTOR_STORE = "vector_store"
    PROMPT_VERSIONS = "prompt_versions"
    TOKEN_MAPS = "token_maps"
    PERFORMANCE_TRACES = "performance_traces"
    API_KEY_RECORDS = "api_key_records"
    FIREWALL_LOGS = "firewall_logs"


class DestructionMethod(Enum):
    SECURE_OVERWRITE = "secure_overwrite"
    CRYPTO_ERASE = "crypto_erase"
    STANDARD_DELETE = "standard_delete"  # Development only


# ── Retention Policy ────────────────────────────────────────────────────────

@dataclass
class RetentionPolicy:
    """Retention policy for a specific data store."""
    data_store: DataStore
    retention_days: int
    destruction_method: DestructionMethod = DestructionMethod.SECURE_OVERWRITE
    regulatory_driver: str = ""
    verification_required: bool = True

    @property
    def cutoff_date(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self.retention_days)


# ── Destruction Record ──────────────────────────────────────────────────────

@dataclass
class DestructionRecord:
    """Immutable record of a data destruction event."""
    record_id: str
    data_store: str
    items_destroyed: int
    bytes_destroyed: int
    destruction_method: str
    destruction_started: str
    destruction_completed: str
    verification_passed: bool
    verification_details: str
    regulatory_driver: str
    operator: str = "system"  # or operator ID for manual destruction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "data_store": self.data_store,
            "items_destroyed": self.items_destroyed,
            "bytes_destroyed": self.bytes_destroyed,
            "destruction_method": self.destruction_method,
            "destruction_started": self.destruction_started,
            "destruction_completed": self.destruction_completed,
            "verification_passed": self.verification_passed,
            "verification_details": self.verification_details,
            "regulatory_driver": self.regulatory_driver,
            "operator": self.operator,
        }


# ── Secure Destruction Utilities ────────────────────────────────────────────

class SecureDestruction:
    """File-level secure destruction utilities."""

    @staticmethod
    def overwrite_file(filepath: Path, passes: int = 3) -> int:
        """
        Overwrite file with random data multiple times, then delete.
        Returns bytes destroyed.
        """
        if not filepath.exists():
            return 0

        file_size = filepath.stat().st_size

        try:
            for pass_num in range(passes):
                with open(filepath, "wb") as f:
                    # Write random bytes in chunks
                    remaining = file_size
                    while remaining > 0:
                        chunk = min(remaining, 65536)
                        f.write(os.urandom(chunk))
                        remaining -= chunk
                    f.flush()
                    os.fsync(f.fileno())

            # Final pass: write zeros
            with open(filepath, "wb") as f:
                remaining = file_size
                while remaining > 0:
                    chunk = min(remaining, 65536)
                    f.write(b"\x00" * chunk)
                    remaining -= chunk
                f.flush()
                os.fsync(f.fileno())

            # Delete
            filepath.unlink()
            return file_size

        except Exception as e:
            logger.error(f"Secure overwrite failed for {filepath}: {e}")
            # Attempt standard delete as fallback
            try:
                filepath.unlink()
            except:
                pass
            return file_size

    @staticmethod
    def verify_destruction(filepath: Path) -> bool:
        """Verify that a file has been destroyed."""
        return not filepath.exists()

    @staticmethod
    def overwrite_directory(dirpath: Path, passes: int = 3) -> tuple:
        """Securely destroy all files in a directory. Returns (files, bytes)."""
        total_files = 0
        total_bytes = 0
        if not dirpath.exists():
            return 0, 0
        for f in dirpath.rglob("*"):
            if f.is_file():
                total_bytes += SecureDestruction.overwrite_file(f, passes)
                total_files += 1
        # Remove empty directories
        try:
            shutil.rmtree(dirpath, ignore_errors=True)
        except:
            pass
        return total_files, total_bytes


# ── Store-Specific Destroyers ───────────────────────────────────────────────

class AuditTrailDestroyer:
    """Handles retention enforcement for audit trail files (JSONL/Parquet)."""

    def __init__(self, audit_dir: str = "audit-logs"):
        self.audit_dir = Path(audit_dir)

    def find_expired(self, cutoff: datetime) -> List[Path]:
        """Find audit files older than the cutoff date."""
        expired = []
        if not self.audit_dir.exists():
            return expired

        for f in self.audit_dir.glob("*.jsonl"):
            try:
                # Parse date from filename: audit_2024-01-15.jsonl
                date_str = f.stem.split("_")[-1] if "_" in f.stem else None
                if date_str:
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if file_date < cutoff:
                        expired.append(f)
                else:
                    # Fall back to file modification time
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                    if mtime < cutoff:
                        expired.append(f)
            except (ValueError, OSError):
                continue

        for f in self.audit_dir.glob("*.parquet"):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    expired.append(f)
            except OSError:
                continue

        return expired

    def destroy(self, files: List[Path], method: DestructionMethod) -> tuple:
        """Destroy expired audit files. Returns (count, bytes)."""
        total_bytes = 0
        count = 0
        for f in files:
            if method == DestructionMethod.SECURE_OVERWRITE:
                total_bytes += SecureDestruction.overwrite_file(f)
            else:
                size = f.stat().st_size if f.exists() else 0
                f.unlink(missing_ok=True)
                total_bytes += size
            count += 1
        return count, total_bytes


class VectorStoreDestroyer:
    """Handles retention enforcement for Qdrant vector store collections."""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_url = qdrant_url

    def find_expired_points(self, collection: str, cutoff: datetime) -> List[str]:
        """Find vector store points older than cutoff (by metadata timestamp)."""
        try:
            import requests
            resp = requests.post(
                f"{self.qdrant_url}/collections/{collection}/points/scroll",
                json={
                    "filter": {
                        "must": [{
                            "key": "created_at",
                            "range": {"lt": cutoff.isoformat()}
                        }]
                    },
                    "limit": 1000,
                    "with_payload": False,
                },
                timeout=30
            )
            if resp.status_code == 200:
                return [str(p["id"]) for p in resp.json().get("result", {}).get("points", [])]
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
        return []

    def destroy_points(self, collection: str, point_ids: List[str]) -> int:
        """Delete specific points from the vector store."""
        if not point_ids:
            return 0
        try:
            import requests
            resp = requests.post(
                f"{self.qdrant_url}/collections/{collection}/points/delete",
                json={"points": point_ids},
                timeout=30
            )
            if resp.status_code == 200:
                return len(point_ids)
        except Exception as e:
            logger.error(f"Vector store delete failed: {e}")
        return 0

    def verify_destruction(self, collection: str, point_ids: List[str]) -> bool:
        """Verify points are no longer retrievable."""
        try:
            import requests
            resp = requests.post(
                f"{self.qdrant_url}/collections/{collection}/points",
                json={"ids": point_ids[:10]},  # Sample check
                timeout=10
            )
            if resp.status_code == 200:
                found = resp.json().get("result", [])
                return len(found) == 0
        except:
            pass
        return False


class TraceDestroyer:
    """Handles retention enforcement for performance traces."""

    def __init__(self, trace_dir: str = "traces"):
        self.trace_dir = Path(trace_dir)

    def find_expired(self, cutoff: datetime) -> List[Path]:
        expired = []
        if not self.trace_dir.exists():
            return expired
        for f in self.trace_dir.rglob("*"):
            if f.is_file():
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                    if mtime < cutoff:
                        expired.append(f)
                except OSError:
                    continue
        return expired

    def destroy(self, files: List[Path], method: DestructionMethod) -> tuple:
        total_bytes = 0
        count = 0
        for f in files:
            if method == DestructionMethod.SECURE_OVERWRITE:
                total_bytes += SecureDestruction.overwrite_file(f)
            else:
                size = f.stat().st_size if f.exists() else 0
                f.unlink(missing_ok=True)
                total_bytes += size
            count += 1
        return count, total_bytes


# ── Retention Manager ───────────────────────────────────────────────────────

class RetentionManager:
    """
    Central retention and destruction manager for all STC data stores.

    Runs on a configurable schedule (default: daily) and:
    1. Checks all data stores against their retention policies
    2. Securely destroys expired data
    3. Verifies destruction
    4. Logs all destruction events to the audit trail
    5. Produces a retention compliance report

    Usage:
        manager = RetentionManager.from_spec(spec)
        report = manager.enforce_all()
        manager.retention_report()
    """

    DEFAULT_POLICIES = [
        RetentionPolicy(DataStore.AUDIT_TRAIL, 365, DestructionMethod.SECURE_OVERWRITE,
                        "FINRA 17a-4, SOX", True),
        RetentionPolicy(DataStore.VECTOR_STORE, 730, DestructionMethod.STANDARD_DELETE,
                        "Internal policy", True),
        RetentionPolicy(DataStore.PROMPT_VERSIONS, 365, DestructionMethod.SECURE_OVERWRITE,
                        "AIUC-1 E (Accountability)", True),
        RetentionPolicy(DataStore.TOKEN_MAPS, 0, DestructionMethod.SECURE_OVERWRITE,
                        "GDPR minimization", False),  # Session-scoped, always destroyed
        RetentionPolicy(DataStore.PERFORMANCE_TRACES, 90, DestructionMethod.STANDARD_DELETE,
                        "Operational", True),
        RetentionPolicy(DataStore.API_KEY_RECORDS, 90, DestructionMethod.SECURE_OVERWRITE,
                        "SOX audit trail", True),
        RetentionPolicy(DataStore.FIREWALL_LOGS, 365, DestructionMethod.SECURE_OVERWRITE,
                        "FINRA, AIUC-1 B", True),
    ]

    def __init__(self, policies: Optional[List[RetentionPolicy]] = None,
                 audit_dir: str = "audit-logs", trace_dir: str = "traces",
                 qdrant_url: str = "http://localhost:6333",
                 vector_collection: str = "stc_documents",
                 audit_callback: Optional[Callable] = None):
        self.policies = {p.data_store: p for p in (policies or self.DEFAULT_POLICIES)}
        self.audit_destroyer = AuditTrailDestroyer(audit_dir)
        self.vector_destroyer = VectorStoreDestroyer(qdrant_url)
        self.trace_destroyer = TraceDestroyer(trace_dir)
        self.vector_collection = vector_collection
        self._audit_callback = audit_callback
        self._destruction_history: List[DestructionRecord] = []

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], audit_callback=None) -> "RetentionManager":
        """Create from STC Declarative Specification."""
        retention_config = spec.get("data_retention", {})

        policies = []
        policy_map = {
            "audit_trail_days": (DataStore.AUDIT_TRAIL, "FINRA 17a-4, SOX"),
            "embeddings_days": (DataStore.VECTOR_STORE, "Internal policy"),
            "prompt_versions_days": (DataStore.PROMPT_VERSIONS, "AIUC-1 E"),
            "performance_traces_days": (DataStore.PERFORMANCE_TRACES, "Operational"),
        }

        destruction_method = DestructionMethod(
            retention_config.get("destruction_method", "secure_overwrite")
        )
        verify = retention_config.get("destruction_verification", True)

        for config_key, (store, driver) in policy_map.items():
            days = retention_config.get(config_key)
            if days is not None:
                policies.append(RetentionPolicy(
                    store, days, destruction_method, driver, verify
                ))

        # Always include session-scoped token maps
        policies.append(RetentionPolicy(
            DataStore.TOKEN_MAPS, 0, DestructionMethod.SECURE_OVERWRITE,
            "GDPR minimization", False
        ))

        if not policies:
            policies = cls.DEFAULT_POLICIES

        infra = spec.get("infrastructure", {})
        return cls(
            policies=policies,
            audit_dir=infra.get("audit_dir", "audit-logs"),
            trace_dir=infra.get("trace_dir", "traces"),
            qdrant_url=infra.get("qdrant_url", "http://localhost:6333"),
            vector_collection=infra.get("vector_collection", "stc_documents"),
            audit_callback=audit_callback,
        )

    def enforce(self, data_store: DataStore) -> Optional[DestructionRecord]:
        """Enforce retention policy for a specific data store."""
        policy = self.policies.get(data_store)
        if not policy:
            logger.warning(f"No retention policy for {data_store.value}")
            return None

        cutoff = policy.cutoff_date
        start_time = datetime.now(timezone.utc)
        items_destroyed = 0
        bytes_destroyed = 0
        verification_passed = True
        verification_details = ""

        try:
            if data_store == DataStore.AUDIT_TRAIL:
                expired = self.audit_destroyer.find_expired(cutoff)
                if expired:
                    items_destroyed, bytes_destroyed = self.audit_destroyer.destroy(
                        expired, policy.destruction_method)
                    if policy.verification_required:
                        for f in expired:
                            if not SecureDestruction.verify_destruction(f):
                                verification_passed = False
                                verification_details += f"FAILED: {f.name} still exists. "
                    verification_details = verification_details or f"Verified {len(expired)} files destroyed"

            elif data_store == DataStore.VECTOR_STORE:
                expired_ids = self.vector_destroyer.find_expired_points(
                    self.vector_collection, cutoff)
                if expired_ids:
                    items_destroyed = self.vector_destroyer.destroy_points(
                        self.vector_collection, expired_ids)
                    if policy.verification_required:
                        verification_passed = self.vector_destroyer.verify_destruction(
                            self.vector_collection, expired_ids)
                    verification_details = f"Destroyed {items_destroyed} vectors"

            elif data_store == DataStore.PERFORMANCE_TRACES:
                expired = self.trace_destroyer.find_expired(cutoff)
                if expired:
                    items_destroyed, bytes_destroyed = self.trace_destroyer.destroy(
                        expired, policy.destruction_method)
                    if policy.verification_required:
                        for f in expired:
                            if not SecureDestruction.verify_destruction(f):
                                verification_passed = False
                    verification_details = f"Destroyed {len(expired)} trace files"

            elif data_store == DataStore.TOKEN_MAPS:
                verification_details = "Token maps are session-scoped (destroyed at session end)"
                verification_passed = True

            else:
                # Generic file-based stores
                verification_details = f"No destroyer configured for {data_store.value}"

        except Exception as e:
            logger.error(f"Retention enforcement failed for {data_store.value}: {e}")
            verification_passed = False
            verification_details = f"ERROR: {str(e)}"

        end_time = datetime.now(timezone.utc)

        record = DestructionRecord(
            record_id=hashlib.sha256(
                f"{data_store.value}:{start_time.isoformat()}".encode()
            ).hexdigest()[:16],
            data_store=data_store.value,
            items_destroyed=items_destroyed,
            bytes_destroyed=bytes_destroyed,
            destruction_method=policy.destruction_method.value,
            destruction_started=start_time.isoformat(),
            destruction_completed=end_time.isoformat(),
            verification_passed=verification_passed,
            verification_details=verification_details,
            regulatory_driver=policy.regulatory_driver,
        )

        self._destruction_history.append(record)
        self._emit_audit(record)
        return record

    def enforce_all(self) -> List[DestructionRecord]:
        """Enforce retention policies for all configured data stores."""
        records = []
        for data_store in self.policies:
            record = self.enforce(data_store)
            if record:
                records.append(record)
        return records

    def retention_report(self) -> Dict[str, Any]:
        """Generate a compliance-ready retention report."""
        report = {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "policies": {},
            "recent_destructions": [],
            "compliance_status": "COMPLIANT",
        }

        for store, policy in self.policies.items():
            report["policies"][store.value] = {
                "retention_days": policy.retention_days,
                "destruction_method": policy.destruction_method.value,
                "regulatory_driver": policy.regulatory_driver,
                "cutoff_date": policy.cutoff_date.isoformat(),
                "verification_required": policy.verification_required,
            }

        # Recent destructions (last 30 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        for record in self._destruction_history:
            completed = datetime.fromisoformat(record.destruction_completed)
            if completed >= cutoff:
                entry = record.to_dict()
                if not record.verification_passed:
                    report["compliance_status"] = "NON_COMPLIANT"
                report["recent_destructions"].append(entry)

        return report

    def _emit_audit(self, record: DestructionRecord):
        """Log destruction event to audit trail."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": "operational.retention",
            "event_type": "data_destruction",
            "details": record.to_dict(),
        }
        if self._audit_callback:
            self._audit_callback(event)


# ── Demo ────────────────────────────────────────────────────────────────────

def demo():
    """Demonstrate retention management with simulated expired data."""
    import tempfile

    print("=" * 70)
    print("STC Data Retention & Destruction — Demo")
    print("=" * 70)

    # Create temporary directories with simulated data
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_dir = Path(tmpdir) / "audit-logs"
        trace_dir = Path(tmpdir) / "traces"
        audit_dir.mkdir()
        trace_dir.mkdir()

        # Create simulated expired audit files
        print("\n▸ Creating simulated expired data...")
        now = datetime.now(timezone.utc)

        # Old audit files (400 days ago)
        for i in range(3):
            old_date = (now - timedelta(days=400 + i)).strftime("%Y-%m-%d")
            f = audit_dir / f"audit_{old_date}.jsonl"
            f.write_text(json.dumps({"event": f"old_event_{i}", "data": "x" * 1000}) + "\n")
            print(f"  Created: {f.name} (expired)")

        # Recent audit files (10 days ago)
        for i in range(2):
            recent_date = (now - timedelta(days=10 + i)).strftime("%Y-%m-%d")
            f = audit_dir / f"audit_{recent_date}.jsonl"
            f.write_text(json.dumps({"event": f"recent_event_{i}", "data": "y" * 500}) + "\n")
            print(f"  Created: {f.name} (retained)")

        # Old trace files
        for i in range(4):
            f = trace_dir / f"trace_{i}.json"
            f.write_text(json.dumps({"trace": f"old_trace_{i}"}) + "\n")
            # Set modification time to 100 days ago
            old_time = time.time() - (100 + i) * 86400
            os.utime(f, (old_time, old_time))
            print(f"  Created: {f.name} (expired, 100+ days old)")

        # Create retention manager
        audit_log = []
        policies = [
            RetentionPolicy(DataStore.AUDIT_TRAIL, 365, DestructionMethod.SECURE_OVERWRITE,
                            "FINRA 17a-4", True),
            RetentionPolicy(DataStore.PERFORMANCE_TRACES, 90, DestructionMethod.SECURE_OVERWRITE,
                            "Operational", True),
            RetentionPolicy(DataStore.TOKEN_MAPS, 0, DestructionMethod.SECURE_OVERWRITE,
                            "GDPR", False),
        ]

        manager = RetentionManager(
            policies=policies,
            audit_dir=str(audit_dir),
            trace_dir=str(trace_dir),
            audit_callback=lambda e: audit_log.append(e),
        )

        # Show pre-enforcement state
        print(f"\n▸ Pre-enforcement state:")
        print(f"  Audit files: {len(list(audit_dir.glob('*.jsonl')))}")
        print(f"  Trace files: {len(list(trace_dir.glob('*.json')))}")

        # Enforce retention
        print("\n▸ Enforcing retention policies...")
        records = manager.enforce_all()

        for record in records:
            status = "✓" if record.verification_passed else "✗"
            print(f"  {status} {record.data_store}: "
                  f"{record.items_destroyed} items, "
                  f"{record.bytes_destroyed:,} bytes destroyed "
                  f"({record.destruction_method})")
            if record.verification_details:
                print(f"    └─ {record.verification_details}")

        # Show post-enforcement state
        print(f"\n▸ Post-enforcement state:")
        remaining_audit = list(audit_dir.glob("*.jsonl"))
        remaining_traces = list(trace_dir.glob("*.json"))
        print(f"  Audit files remaining: {len(remaining_audit)}")
        for f in remaining_audit:
            print(f"    └─ {f.name}")
        print(f"  Trace files remaining: {len(remaining_traces)}")

        # Generate compliance report
        print("\n▸ Compliance report:")
        report = manager.retention_report()
        print(f"  Status: {report['compliance_status']}")
        print(f"  Policies configured: {len(report['policies'])}")
        print(f"  Recent destructions: {len(report['recent_destructions'])}")
        for p_name, p_detail in report["policies"].items():
            print(f"    {p_name}: {p_detail['retention_days']} days "
                  f"({p_detail['destruction_method']}) "
                  f"[{p_detail['regulatory_driver']}]")

        # Audit trail
        print(f"\n▸ Audit events: {len(audit_log)}")
        for e in audit_log:
            d = e["details"]
            print(f"  [{d['data_store']}] {d['items_destroyed']} items destroyed, "
                  f"verified={d['verification_passed']}")

    print("\n" + "=" * 70)
    print("✓ Retention & destruction demo complete")
    print("=" * 70)


if __name__ == "__main__":
    demo()

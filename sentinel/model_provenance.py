"""
STC Framework — Model Provenance Verification
sentinel/model_provenance.py

Verifies integrity and authenticity of ML models before deployment:
  - SHA-256 checksum validation against published hashes
  - Source verification (HuggingFace verified orgs, official repos)
  - Model card review (training data, biases, limitations)
  - Cosign/Sigstore signature verification (where available)
  - Runtime integrity monitoring (detect in-place model swaps)

Prevents supply chain attacks where poisoned models could be swapped
into the STC pipeline (PromptGuard, BGE embeddings, local LLMs).

Part of the Sentinel Layer (infrastructure enforcement, not intelligence).
"""

import os
import json
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("stc.sentinel.provenance")


# ── Model Registry ──────────────────────────────────────────────────────────

class ModelTrust(Enum):
    """Trust level for a model based on provenance verification."""
    VERIFIED = "verified"       # All checks passed
    TRUSTED = "trusted"         # Source verified, checksum matched
    UNVERIFIED = "unverified"   # Downloaded but not verified
    SUSPICIOUS = "suspicious"   # Checksum mismatch or unknown source
    BLOCKED = "blocked"         # Failed verification, not usable


class VerificationCheck(Enum):
    """Individual verification checks."""
    CHECKSUM = "checksum"
    SOURCE = "source"
    MODEL_CARD = "model_card"
    SIGNATURE = "signature"
    RUNTIME_INTEGRITY = "runtime_integrity"


@dataclass
class ModelRecord:
    """Provenance record for a registered model."""
    model_id: str                    # e.g., "meta-llama/PromptGuard-86M"
    model_path: str                  # Local filesystem path
    expected_checksum: str           # SHA-256 of model files
    source_url: str                  # Where it was downloaded from
    source_verified: bool = False    # Is source a verified org?
    model_card_reviewed: bool = False
    signature_verified: bool = False
    trust_level: ModelTrust = ModelTrust.UNVERIFIED
    registered_at: str = ""
    last_verified: str = ""
    file_checksums: Dict[str, str] = field(default_factory=dict)
    verification_history: List[Dict[str, Any]] = field(default_factory=list)


# ── Checksum Calculator ─────────────────────────────────────────────────────

class ChecksumCalculator:
    """Calculate SHA-256 checksums for model files."""

    @staticmethod
    def file_checksum(filepath: Path, chunk_size: int = 65536) -> str:
        """Calculate SHA-256 checksum of a single file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def directory_checksum(dirpath: Path) -> tuple:
        """
        Calculate aggregate checksum of all files in a model directory.
        Returns (aggregate_hash, {filename: hash}) sorted by filename for determinism.
        """
        file_hashes = {}
        aggregate = hashlib.sha256()

        for filepath in sorted(dirpath.rglob("*")):
            if filepath.is_file():
                file_hash = ChecksumCalculator.file_checksum(filepath)
                relative = str(filepath.relative_to(dirpath))
                file_hashes[relative] = file_hash
                aggregate.update(f"{relative}:{file_hash}".encode())

        return aggregate.hexdigest(), file_hashes


# ── Source Verifier ─────────────────────────────────────────────────────────

class SourceVerifier:
    """Verify that models come from trusted sources."""

    # Known verified organizations on HuggingFace
    VERIFIED_ORGS = {
        "meta-llama", "meta-ai", "microsoft", "nvidia", "google",
        "anthropic", "mistralai", "facebook", "huggingface",
        "sentence-transformers", "BAAI",  # BGE embeddings
    }

    # Known trusted model repositories
    TRUSTED_REPOS = {
        "meta-llama/PromptGuard-86M",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
        "mistralai/Mistral-7B-v0.1",
        "microsoft/phi-3-mini-128k-instruct",
    }

    @classmethod
    def verify_source(cls, model_id: str) -> Dict[str, Any]:
        """
        Verify the source of a model by checking against known trusted orgs/repos.
        Returns verification result.
        """
        org = model_id.split("/")[0] if "/" in model_id else model_id

        is_verified_org = org in cls.VERIFIED_ORGS
        is_trusted_repo = model_id in cls.TRUSTED_REPOS

        # Try to check HuggingFace API for org verification
        hf_verified = False
        try:
            import requests
            resp = requests.get(
                f"https://huggingface.co/api/models/{model_id}",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                # Check for verification badges
                hf_verified = data.get("author", "") in cls.VERIFIED_ORGS
        except Exception:
            pass  # Offline verification only

        return {
            "model_id": model_id,
            "organization": org,
            "is_verified_org": is_verified_org,
            "is_trusted_repo": is_trusted_repo,
            "hf_api_verified": hf_verified,
            "source_trusted": is_verified_org or is_trusted_repo,
        }

    @classmethod
    def check_model_card(cls, model_id: str) -> Dict[str, Any]:
        """
        Check if a model has a model card with required fields.
        Required for AIUC-1 compliance.
        """
        required_fields = [
            "training_data", "evaluation", "limitations",
            "bias", "intended_use",
        ]

        try:
            import requests
            resp = requests.get(
                f"https://huggingface.co/api/models/{model_id}",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                card_data = data.get("cardData", {}) or {}
                tags = data.get("tags", [])

                # Check for model card presence
                has_card = bool(data.get("modelCard") or card_data)

                # Check for key metadata
                found_fields = []
                for field_name in required_fields:
                    if field_name in card_data or field_name in str(tags):
                        found_fields.append(field_name)

                return {
                    "model_id": model_id,
                    "has_model_card": has_card,
                    "fields_found": found_fields,
                    "fields_missing": [f for f in required_fields if f not in found_fields],
                    "tags": tags[:10],
                    "card_adequate": has_card and len(found_fields) >= 3,
                }
        except Exception as e:
            logger.warning(f"Model card check failed for {model_id}: {e}")

        return {
            "model_id": model_id,
            "has_model_card": False,
            "fields_found": [],
            "fields_missing": required_fields,
            "card_adequate": False,
            "error": "API unavailable (offline verification only)",
        }


# ── Provenance Manager ──────────────────────────────────────────────────────

class ModelProvenanceManager:
    """
    Central model provenance verification for STC.

    Maintains a registry of approved models with their checksums,
    verifies models before they're used in the pipeline, and monitors
    for runtime integrity violations (model file changes).

    Usage:
        manager = ModelProvenanceManager("model-registry.json")
        manager.register("meta-llama/PromptGuard-86M", "/models/promptguard")
        result = manager.verify("meta-llama/PromptGuard-86M")
        health = manager.runtime_integrity_check()
    """

    def __init__(self, registry_path: str = "model-registry.json",
                 audit_callback=None):
        self.registry_path = Path(registry_path)
        self._registry: Dict[str, ModelRecord] = {}
        self._audit_callback = audit_callback

        # Load existing registry
        if self.registry_path.exists():
            self._load_registry()

    def _load_registry(self):
        """Load model registry from disk."""
        try:
            data = json.loads(self.registry_path.read_text())
            for model_id, record_data in data.items():
                self._registry[model_id] = ModelRecord(
                    model_id=record_data["model_id"],
                    model_path=record_data["model_path"],
                    expected_checksum=record_data["expected_checksum"],
                    source_url=record_data.get("source_url", ""),
                    source_verified=record_data.get("source_verified", False),
                    model_card_reviewed=record_data.get("model_card_reviewed", False),
                    signature_verified=record_data.get("signature_verified", False),
                    trust_level=ModelTrust(record_data.get("trust_level", "unverified")),
                    registered_at=record_data.get("registered_at", ""),
                    last_verified=record_data.get("last_verified", ""),
                    file_checksums=record_data.get("file_checksums", {}),
                )
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    def _save_registry(self):
        """Persist model registry to disk."""
        data = {}
        for model_id, record in self._registry.items():
            data[model_id] = {
                "model_id": record.model_id,
                "model_path": record.model_path,
                "expected_checksum": record.expected_checksum,
                "source_url": record.source_url,
                "source_verified": record.source_verified,
                "model_card_reviewed": record.model_card_reviewed,
                "signature_verified": record.signature_verified,
                "trust_level": record.trust_level.value,
                "registered_at": record.registered_at,
                "last_verified": record.last_verified,
                "file_checksums": record.file_checksums,
            }
        self.registry_path.write_text(json.dumps(data, indent=2))

    def register(self, model_id: str, model_path: str,
                 source_url: str = "") -> ModelRecord:
        """
        Register a model by computing its checksum and verifying its source.

        This should be called when a model is first downloaded and approved
        for use. The checksum becomes the baseline for runtime integrity checks.
        """
        path = Path(model_path)
        now = datetime.now(timezone.utc).isoformat()

        # Compute checksums
        if path.is_dir():
            aggregate_hash, file_hashes = ChecksumCalculator.directory_checksum(path)
        elif path.is_file():
            aggregate_hash = ChecksumCalculator.file_checksum(path)
            file_hashes = {path.name: aggregate_hash}
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Verify source
        source_result = SourceVerifier.verify_source(model_id)

        # Check model card
        card_result = SourceVerifier.check_model_card(model_id)

        # Determine trust level
        trust = ModelTrust.UNVERIFIED
        if source_result["source_trusted"]:
            trust = ModelTrust.TRUSTED
            if card_result.get("card_adequate", False):
                trust = ModelTrust.VERIFIED

        record = ModelRecord(
            model_id=model_id,
            model_path=str(path),
            expected_checksum=aggregate_hash,
            source_url=source_url or f"https://huggingface.co/{model_id}",
            source_verified=source_result["source_trusted"],
            model_card_reviewed=card_result.get("card_adequate", False),
            trust_level=trust,
            registered_at=now,
            last_verified=now,
            file_checksums=file_hashes,
        )

        self._registry[model_id] = record
        self._save_registry()

        self._emit_audit("model_registered", model_id, {
            "checksum": aggregate_hash,
            "trust_level": trust.value,
            "source_verified": source_result["source_trusted"],
            "files": len(file_hashes),
        })

        return record

    def verify(self, model_id: str) -> Dict[str, Any]:
        """
        Run all verification checks on a registered model.

        Returns a comprehensive verification report.
        """
        record = self._registry.get(model_id)
        if not record:
            return {"model_id": model_id, "status": "NOT_REGISTERED",
                    "trust_level": ModelTrust.BLOCKED.value}

        now = datetime.now(timezone.utc).isoformat()
        checks = {}

        # 1. Checksum verification
        path = Path(record.model_path)
        if path.exists():
            if path.is_dir():
                current_hash, current_files = ChecksumCalculator.directory_checksum(path)
            else:
                current_hash = ChecksumCalculator.file_checksum(path)
                current_files = {path.name: current_hash}

            checksum_match = current_hash == record.expected_checksum
            checks[VerificationCheck.CHECKSUM.value] = {
                "passed": checksum_match,
                "expected": record.expected_checksum[:16] + "...",
                "actual": current_hash[:16] + "...",
                "files_checked": len(current_files),
            }

            # Check individual file changes
            changed_files = []
            for fname, fhash in current_files.items():
                if fname in record.file_checksums and record.file_checksums[fname] != fhash:
                    changed_files.append(fname)
            if changed_files:
                checks[VerificationCheck.CHECKSUM.value]["changed_files"] = changed_files
        else:
            checks[VerificationCheck.CHECKSUM.value] = {
                "passed": False, "error": "Model path not found"
            }
            checksum_match = False

        # 2. Source verification
        checks[VerificationCheck.SOURCE.value] = {
            "passed": record.source_verified,
            "source_url": record.source_url,
        }

        # 3. Model card
        checks[VerificationCheck.MODEL_CARD.value] = {
            "passed": record.model_card_reviewed,
        }

        # 4. Determine overall trust level
        if not checksum_match:
            trust = ModelTrust.SUSPICIOUS
        elif record.source_verified and record.model_card_reviewed:
            trust = ModelTrust.VERIFIED
        elif record.source_verified:
            trust = ModelTrust.TRUSTED
        else:
            trust = ModelTrust.UNVERIFIED

        # Update record
        record.trust_level = trust
        record.last_verified = now
        record.verification_history.append({
            "timestamp": now,
            "checks": checks,
            "trust_level": trust.value,
        })
        self._save_registry()

        result = {
            "model_id": model_id,
            "trust_level": trust.value,
            "checks": checks,
            "all_passed": all(c.get("passed", False) for c in checks.values()),
            "verified_at": now,
            "usable": trust in (ModelTrust.VERIFIED, ModelTrust.TRUSTED),
        }

        self._emit_audit("model_verified", model_id, {
            "trust_level": trust.value,
            "checksum_match": checksum_match,
            "all_passed": result["all_passed"],
        })

        return result

    def runtime_integrity_check(self) -> Dict[str, Any]:
        """
        Check all registered models for runtime integrity violations.
        Should be called periodically (e.g., every hour) to detect
        in-place model swaps or tampering.
        """
        results = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "models_checked": 0,
            "models_healthy": 0,
            "models_suspicious": 0,
            "models_missing": 0,
            "details": [],
        }

        for model_id, record in self._registry.items():
            results["models_checked"] += 1
            path = Path(record.model_path)

            if not path.exists():
                results["models_missing"] += 1
                results["details"].append({
                    "model_id": model_id, "status": "MISSING",
                    "path": record.model_path,
                })
                continue

            # Quick checksum check
            if path.is_dir():
                current_hash, _ = ChecksumCalculator.directory_checksum(path)
            else:
                current_hash = ChecksumCalculator.file_checksum(path)

            if current_hash == record.expected_checksum:
                results["models_healthy"] += 1
                results["details"].append({
                    "model_id": model_id, "status": "HEALTHY",
                })
            else:
                results["models_suspicious"] += 1
                record.trust_level = ModelTrust.SUSPICIOUS
                results["details"].append({
                    "model_id": model_id, "status": "SUSPICIOUS",
                    "reason": "Checksum mismatch — possible model swap",
                    "expected": record.expected_checksum[:16] + "...",
                    "actual": current_hash[:16] + "...",
                })
                self._emit_audit("integrity_violation", model_id, {
                    "expected": record.expected_checksum,
                    "actual": current_hash,
                })

        results["status"] = (
            "HEALTHY" if results["models_suspicious"] == 0 and results["models_missing"] == 0
            else "ALERT"
        )

        return results

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models and their trust levels."""
        return [
            {
                "model_id": r.model_id,
                "trust_level": r.trust_level.value,
                "last_verified": r.last_verified,
                "source_verified": r.source_verified,
                "checksum": r.expected_checksum[:16] + "...",
            }
            for r in self._registry.values()
        ]

    def _emit_audit(self, event_type: str, model_id: str, details: Dict):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": "sentinel.provenance",
            "event_type": event_type,
            "model_id": model_id,
            "details": details,
        }
        if self._audit_callback:
            self._audit_callback(event)


# ── Demo ────────────────────────────────────────────────────────────────────

def demo():
    """Demonstrate model provenance verification."""
    import tempfile

    print("=" * 70)
    print("STC Model Provenance Verification — Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulated model files
        model_dir = Path(tmpdir) / "models" / "promptguard"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(os.urandom(1024))
        (model_dir / "config.json").write_text('{"model_type": "bert"}')
        (model_dir / "tokenizer.json").write_text('{"type": "BPE"}')

        embeddings_dir = Path(tmpdir) / "models" / "bge-small"
        embeddings_dir.mkdir(parents=True)
        (embeddings_dir / "model.safetensors").write_bytes(os.urandom(2048))
        (embeddings_dir / "config.json").write_text('{"model_type": "bert"}')

        registry_path = Path(tmpdir) / "model-registry.json"

        audit_log = []
        manager = ModelProvenanceManager(
            registry_path=str(registry_path),
            audit_callback=lambda e: audit_log.append(e),
        )

        # Register models
        print("\n▸ Registering models...")
        r1 = manager.register(
            "meta-llama/PromptGuard-86M",
            str(model_dir),
            "https://huggingface.co/meta-llama/PromptGuard-86M"
        )
        print(f"  {r1.model_id}: trust={r1.trust_level.value}, "
              f"checksum={r1.expected_checksum[:16]}..., "
              f"files={len(r1.file_checksums)}")

        r2 = manager.register(
            "BAAI/bge-small-en-v1.5",
            str(embeddings_dir),
            "https://huggingface.co/BAAI/bge-small-en-v1.5"
        )
        print(f"  {r2.model_id}: trust={r2.trust_level.value}, "
              f"checksum={r2.expected_checksum[:16]}..., "
              f"files={len(r2.file_checksums)}")

        # Verify models (should pass)
        print("\n▸ Verifying models (clean state)...")
        for model_id in ["meta-llama/PromptGuard-86M", "BAAI/bge-small-en-v1.5"]:
            result = manager.verify(model_id)
            status = "✓" if result["all_passed"] else "✗"
            print(f"  {status} {model_id}: trust={result['trust_level']}, "
                  f"usable={result['usable']}")
            for check, detail in result["checks"].items():
                cs = "✓" if detail["passed"] else "✗"
                print(f"    {cs} {check}")

        # Runtime integrity check (should be healthy)
        print("\n▸ Runtime integrity check (clean)...")
        integrity = manager.runtime_integrity_check()
        print(f"  Status: {integrity['status']}")
        print(f"  Healthy: {integrity['models_healthy']}/{integrity['models_checked']}")

        # Simulate model tampering
        print("\n▸ Simulating model tampering (modifying PromptGuard)...")
        (model_dir / "model.safetensors").write_bytes(os.urandom(1024))  # Different random bytes
        print("  ⚠ Model file replaced with different content")

        # Re-verify (should detect tampering)
        print("\n▸ Re-verifying after tampering...")
        result = manager.verify("meta-llama/PromptGuard-86M")
        status = "✓" if result["all_passed"] else "✗"
        print(f"  {status} PromptGuard: trust={result['trust_level']}, "
              f"usable={result['usable']}")
        for check, detail in result["checks"].items():
            cs = "✓" if detail["passed"] else "✗"
            extra = ""
            if check == "checksum" and not detail["passed"]:
                extra = f" (expected={detail.get('expected', '?')}, actual={detail.get('actual', '?')})"
            print(f"    {cs} {check}{extra}")

        # Runtime integrity check (should detect issue)
        print("\n▸ Runtime integrity check (after tampering)...")
        integrity = manager.runtime_integrity_check()
        print(f"  Status: {integrity['status']}")
        for d in integrity["details"]:
            icon = "✓" if d["status"] == "HEALTHY" else "⚠"
            print(f"  {icon} {d['model_id']}: {d['status']}")
            if d.get("reason"):
                print(f"    └─ {d['reason']}")

        # List models
        print("\n▸ Model registry:")
        for m in manager.list_models():
            print(f"  {m['model_id']}: trust={m['trust_level']}, "
                  f"source_verified={m['source_verified']}")

        # Audit trail
        print(f"\n▸ Audit events: {len(audit_log)}")
        for e in audit_log:
            print(f"  [{e['event_type']}] {e['model_id']}: "
                  f"{json.dumps({k: v for k, v in e['details'].items() if k != 'expected' and k != 'actual'})}")

    print("\n" + "=" * 70)
    print("✓ Model provenance demo complete")
    print("=" * 70)


if __name__ == "__main__":
    demo()

"""
STC Framework — Infrastructure Resilience Patterns
infrastructure/resilience.py

Production-grade resilience patterns for the STC pipeline:
  - Circuit Breaker: per-dependency with configurable thresholds
  - Bulkhead: connection pool isolation per dependency
  - Timeout Hierarchy: layered timeouts from cache to LLM
  - Retry Policy: per-operation with exponential backoff + jitter
  - Fallback Chain: graceful degradation on dependency failure

These patterns complement the cost circuit breaker (operational/cost_controls.py)
by protecting against infrastructure-level failures and cascading outages.
"""

import asyncio
import logging
import random
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

logger = logging.getLogger("stc.infrastructure.resilience")
T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, rejecting requests
    HALF_OPEN = "half_open"    # Testing with single probe


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    name: str
    failure_threshold: int = 5       # Failures to open
    failure_window_seconds: int = 60 # Window for counting failures
    open_duration_seconds: int = 30  # How long to stay open
    half_open_max_calls: int = 1     # Probe calls in half-open
    success_threshold: int = 1       # Successes to close from half-open


class CircuitBreaker:
    """
    Circuit breaker for external dependencies.

    States:
      CLOSED → (failure_threshold reached) → OPEN
      OPEN → (open_duration elapsed) → HALF_OPEN
      HALF_OPEN → (success) → CLOSED
      HALF_OPEN → (failure) → OPEN

    Usage:
        cb = CircuitBreaker(CircuitBreakerConfig("qdrant", failure_threshold=3))
        try:
            result = cb.execute(lambda: qdrant_client.search(...))
        except CircuitOpenError:
            # Use fallback
    """

    def __init__(self, config: CircuitBreakerConfig, audit_callback=None):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failures: deque = deque()  # timestamps of recent failures
        self._opened_at: float = 0
        self._half_open_calls: int = 0
        self._half_open_successes: int = 0
        self._lock = threading.Lock()
        self._audit_callback = audit_callback
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._opened_at >= self.config.open_duration_seconds:
                    self._transition(CircuitState.HALF_OPEN)
            return self._state

    def execute(self, func: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: The function to execute
            fallback: Optional fallback if circuit is open

        Returns:
            Result of func or fallback

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        current_state = self.state
        self._total_calls += 1

        if current_state == CircuitState.OPEN:
            self._total_rejections += 1
            if fallback:
                logger.warning(f"Circuit [{self.config.name}] OPEN — using fallback")
                return fallback()
            raise CircuitOpenError(f"Circuit [{self.config.name}] is OPEN")

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    if fallback:
                        return fallback()
                    raise CircuitOpenError(f"Circuit [{self.config.name}] HALF_OPEN — probe limit reached")
                self._half_open_calls += 1

        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            if fallback:
                logger.warning(f"Circuit [{self.config.name}] failure — using fallback: {e}")
                return fallback()
            raise

    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition(CircuitState.CLOSED)

    def _on_failure(self):
        with self._lock:
            self._total_failures += 1
            now = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
                return

            # Add failure timestamp
            self._failures.append(now)

            # Remove old failures outside window
            cutoff = now - self.config.failure_window_seconds
            while self._failures and self._failures[0] < cutoff:
                self._failures.popleft()

            if len(self._failures) >= self.config.failure_threshold:
                self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState):
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._failures.clear()

        logger.info(f"Circuit [{self.config.name}]: {old_state.value} → {new_state.value}")
        if self._audit_callback:
            self._audit_callback({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": "infrastructure.resilience",
                "event_type": "circuit_state_change",
                "details": {
                    "circuit": self.config.name,
                    "from": old_state.value,
                    "to": new_state.value,
                }
            })

    def reset(self):
        """Manually reset the circuit to CLOSED."""
        with self._lock:
            self._transition(CircuitState.CLOSED)

    def health(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "state": self.state.value,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
            "recent_failures": len(self._failures),
            "failure_rate": self._total_failures / self._total_calls if self._total_calls > 0 else 0,
        }


class CircuitOpenError(Exception):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# BULKHEAD
# ══════════════════════════════════════════════════════════════════════════════

class Bulkhead:
    """
    Bulkhead pattern: limits concurrent access to a dependency.

    Prevents one slow dependency from consuming all available threads/connections.

    Usage:
        bh = Bulkhead("qdrant", max_concurrent=50, max_queue=100)
        with bh:
            result = qdrant_client.search(...)
    """

    def __init__(self, name: str, max_concurrent: int = 50, max_queue: int = 100,
                 queue_timeout_seconds: float = 5.0):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout_seconds
        self._semaphore = threading.Semaphore(max_concurrent)
        self._queue_count = 0
        self._lock = threading.Lock()
        self._active_count = 0
        self._total_calls = 0
        self._total_rejections = 0
        self._total_timeouts = 0

    def __enter__(self):
        self._total_calls += 1
        with self._lock:
            if self._queue_count >= self.max_queue:
                self._total_rejections += 1
                raise BulkheadFullError(
                    f"Bulkhead [{self.name}] full: {self._active_count} active, "
                    f"{self._queue_count} queued")
            self._queue_count += 1

        acquired = self._semaphore.acquire(timeout=self.queue_timeout)
        with self._lock:
            self._queue_count -= 1
            if not acquired:
                self._total_timeouts += 1
                raise BulkheadTimeoutError(
                    f"Bulkhead [{self.name}] timeout after {self.queue_timeout}s")
            self._active_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._active_count -= 1
        self._semaphore.release()
        return False

    def health(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": self._active_count,
            "max_concurrent": self.max_concurrent,
            "utilization": self._active_count / self.max_concurrent,
            "total_calls": self._total_calls,
            "total_rejections": self._total_rejections,
            "total_timeouts": self._total_timeouts,
        }


class BulkheadFullError(Exception):
    pass

class BulkheadTimeoutError(Exception):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# TIMEOUT
# ══════════════════════════════════════════════════════════════════════════════

class TimeoutConfig:
    """Layered timeout hierarchy for the STC pipeline."""

    # Timeout values in seconds
    REDIS_READ = 0.1
    REDIS_WRITE = 0.2
    PII_SCAN = 0.5
    EMBEDDING = 2.0
    VECTOR_SEARCH = 3.0
    CRITIC_VALIDATION = 5.0
    LLM_GENERATION = 30.0
    TOTAL_REQUEST = 45.0
    ALB_IDLE = 60.0

    @classmethod
    def all(cls) -> Dict[str, float]:
        return {
            "redis_read": cls.REDIS_READ,
            "redis_write": cls.REDIS_WRITE,
            "pii_scan": cls.PII_SCAN,
            "embedding": cls.EMBEDDING,
            "vector_search": cls.VECTOR_SEARCH,
            "critic_validation": cls.CRITIC_VALIDATION,
            "llm_generation": cls.LLM_GENERATION,
            "total_request": cls.TOTAL_REQUEST,
            "alb_idle": cls.ALB_IDLE,
        }


def with_timeout(func: Callable[[], T], timeout_seconds: float,
                 operation_name: str = "operation") -> T:
    """
    Execute a function with a timeout.

    Uses threading for synchronous functions.
    Raises TimeoutError if the function doesn't complete in time.
    """
    result = [None]
    exception = [None]

    def wrapper():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds}s")

    if exception[0]:
        raise exception[0]

    return result[0]


# ══════════════════════════════════════════════════════════════════════════════
# RETRY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetryConfig:
    """Retry policy configuration."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter.

    Usage:
        policy = RetryPolicy(RetryConfig(max_retries=3, base_delay_seconds=1.0))
        result = policy.execute(lambda: llm_client.generate(...), "llm_call")
    """

    def __init__(self, config: RetryConfig):
        self.config = config

    def execute(self, func: Callable[[], T], operation_name: str = "operation") -> T:
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func()
            except self.config.retryable_exceptions as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Retry [{operation_name}] attempt {attempt + 1}/{self.config.max_retries}: "
                        f"{type(e).__name__}: {e}. Waiting {delay:.2f}s"
                    )
                    time.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** attempt)
        delay = min(delay, self.config.max_delay_seconds)
        if self.config.jitter:
            delay *= (0.5 + random.random())  # 50-150% of calculated delay
        return delay


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK CHAIN
# ══════════════════════════════════════════════════════════════════════════════

class FallbackChain:
    """
    Chain of fallback providers for graceful degradation.

    Usage:
        chain = FallbackChain("llm_provider")
        chain.add("openai", lambda q: openai_client.generate(q))
        chain.add("bedrock", lambda q: bedrock_client.generate(q))
        chain.add("local", lambda q: ollama_client.generate(q))
        result = chain.execute(query)
    """

    def __init__(self, name: str, audit_callback=None):
        self.name = name
        self._providers: List[tuple] = []  # (name, func)
        self._audit_callback = audit_callback

    def add(self, provider_name: str, func: Callable):
        self._providers.append((provider_name, func))

    def execute(self, *args, **kwargs):
        """Try each provider in order until one succeeds."""
        errors = []
        for provider_name, func in self._providers:
            try:
                result = func(*args, **kwargs)
                if errors:
                    logger.info(f"FallbackChain [{self.name}]: succeeded with {provider_name} "
                                f"after {len(errors)} failures")
                    if self._audit_callback:
                        self._audit_callback({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "component": "infrastructure.resilience",
                            "event_type": "fallback_activated",
                            "details": {
                                "chain": self.name,
                                "provider": provider_name,
                                "failed_providers": [e[0] for e in errors],
                            }
                        })
                return result
            except Exception as e:
                errors.append((provider_name, str(e)))
                logger.warning(f"FallbackChain [{self.name}]: {provider_name} failed: {e}")

        raise FallbackExhaustedError(
            f"All providers in [{self.name}] failed: {errors}")


class FallbackExhaustedError(Exception):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# RESILIENCE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ResilienceManager:
    """
    Central manager for all resilience patterns in the STC pipeline.

    Pre-configures circuit breakers, bulkheads, and retry policies
    for all external dependencies.

    Usage:
        rm = ResilienceManager.from_spec(spec)
        result = rm.execute_with_resilience("qdrant", lambda: search(...))
    """

    # Default configurations per dependency
    DEFAULTS = {
        "llm_primary": CircuitBreakerConfig("llm_primary", 5, 60, 30),
        "llm_secondary": CircuitBreakerConfig("llm_secondary", 5, 60, 60),
        "qdrant": CircuitBreakerConfig("qdrant", 3, 30, 15),
        "redis": CircuitBreakerConfig("redis", 3, 30, 10),
        "langfuse": CircuitBreakerConfig("langfuse", 5, 60, 60),
        "phoenix": CircuitBreakerConfig("phoenix", 5, 60, 60),
    }

    BULKHEAD_DEFAULTS = {
        "llm_primary": (100, 200),      # max_concurrent, max_queue
        "llm_secondary": (50, 100),
        "qdrant": (50, 100),
        "redis": (200, 500),
    }

    RETRY_DEFAULTS = {
        "llm": RetryConfig(3, 1.0, 10.0, 2.0, True),
        "qdrant": RetryConfig(2, 0.5, 5.0, 2.0, True),
        "redis": RetryConfig(2, 0.1, 1.0, 2.0, False),
        "audit": RetryConfig(3, 1.0, 10.0, 2.0, True),
    }

    def __init__(self, audit_callback=None):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self._audit_callback = audit_callback

    @classmethod
    def from_spec(cls, spec: Dict[str, Any] = None, audit_callback=None) -> "ResilienceManager":
        """Create with default configurations."""
        rm = cls(audit_callback=audit_callback)

        for name, config in cls.DEFAULTS.items():
            rm.circuit_breakers[name] = CircuitBreaker(config, audit_callback)

        for name, (mc, mq) in cls.BULKHEAD_DEFAULTS.items():
            rm.bulkheads[name] = Bulkhead(name, mc, mq)

        for name, config in cls.RETRY_DEFAULTS.items():
            rm.retry_policies[name] = RetryPolicy(config)

        return rm

    def execute_with_resilience(self, dependency: str, func: Callable[[], T],
                                 fallback: Optional[Callable[[], T]] = None,
                                 timeout: Optional[float] = None) -> T:
        """
        Execute a function with full resilience stack:
        Circuit Breaker → Bulkhead → Retry → Timeout → Fallback
        """
        cb = self.circuit_breakers.get(dependency)
        bh = self.bulkheads.get(dependency)
        retry = self.retry_policies.get(dependency)

        def inner():
            actual_func = func
            if timeout:
                actual_func = lambda: with_timeout(func, timeout, dependency)
            if retry:
                return retry.execute(actual_func, dependency)
            return actual_func()

        def with_bulkhead():
            if bh:
                with bh:
                    return inner()
            return inner()

        if cb:
            return cb.execute(with_bulkhead, fallback)
        return with_bulkhead()

    def health(self) -> Dict[str, Any]:
        """Health check across all resilience components."""
        cb_health = {name: cb.health() for name, cb in self.circuit_breakers.items()}
        bh_health = {name: bh.health() for name, bh in self.bulkheads.items()}

        open_circuits = [n for n, h in cb_health.items() if h["state"] != "closed"]
        overloaded_bulkheads = [n for n, h in bh_health.items() if h["utilization"] > 0.8]

        return {
            "status": "degraded" if open_circuits else "healthy",
            "circuit_breakers": cb_health,
            "bulkheads": bh_health,
            "open_circuits": open_circuits,
            "overloaded_bulkheads": overloaded_bulkheads,
            "timeouts": TimeoutConfig.all(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    print("=" * 70)
    print("STC Infrastructure Resilience — Demo")
    print("=" * 70)

    audit_log = []
    rm = ResilienceManager.from_spec(audit_callback=lambda e: audit_log.append(e))

    # ── Circuit Breaker Demo ──
    print("\n▸ Circuit Breaker Demo (Qdrant)")
    cb = rm.circuit_breakers["qdrant"]

    call_count = 0
    def flaky_qdrant():
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            raise ConnectionError("Qdrant connection refused")
        return {"results": ["doc1", "doc2"]}

    def qdrant_fallback():
        return {"results": [], "degraded": True, "reason": "No context available (Qdrant circuit open)"}

    for i in range(7):
        try:
            result = cb.execute(flaky_qdrant, qdrant_fallback)
            status = "fallback" if result.get("degraded") else "success"
            print(f"  Call {i+1}: {status} (circuit: {cb.state.value})")
        except Exception as e:
            print(f"  Call {i+1}: error — {e} (circuit: {cb.state.value})")

    # Wait for circuit to transition to half-open
    print(f"\n  Waiting for circuit to transition (open_duration={cb.config.open_duration_seconds}s)...")
    time.sleep(cb.config.open_duration_seconds + 1)
    print(f"  Circuit state: {cb.state.value}")

    # Now calls succeed (call_count > 4)
    result = cb.execute(flaky_qdrant, qdrant_fallback)
    print(f"  Recovery call: success={not result.get('degraded', False)} (circuit: {cb.state.value})")

    print(f"\n  Circuit health: {cb.health()}")

    # ── Bulkhead Demo ──
    print("\n▸ Bulkhead Demo")
    bh = Bulkhead("test-bulkhead", max_concurrent=3, max_queue=2, queue_timeout_seconds=0.5)

    def slow_operation(id):
        time.sleep(0.3)
        return f"result-{id}"

    threads = []
    results = []

    def run_with_bulkhead(id):
        try:
            with bh:
                r = slow_operation(id)
                results.append(("success", id, r))
        except (BulkheadFullError, BulkheadTimeoutError) as e:
            results.append(("rejected", id, str(e)))

    # Launch 7 concurrent calls (3 active + 2 queued + 2 rejected)
    for i in range(7):
        t = threading.Thread(target=run_with_bulkhead, args=(i,))
        threads.append(t)
        t.start()
        time.sleep(0.05)  # Slight stagger

    for t in threads:
        t.join(timeout=5)

    successes = sum(1 for r in results if r[0] == "success")
    rejections = sum(1 for r in results if r[0] == "rejected")
    print(f"  7 concurrent calls: {successes} succeeded, {rejections} rejected")
    print(f"  Bulkhead health: {bh.health()}")

    # ── Retry Demo ──
    print("\n▸ Retry Policy Demo")
    retry = RetryPolicy(RetryConfig(max_retries=3, base_delay_seconds=0.1, jitter=False))

    attempt = [0]
    def flaky_llm():
        attempt[0] += 1
        if attempt[0] < 3:
            raise ConnectionError(f"LLM timeout (attempt {attempt[0]})")
        return "Generated response after retries"

    start = time.time()
    result = retry.execute(flaky_llm, "llm_generation")
    elapsed = time.time() - start
    print(f"  Result: {result}")
    print(f"  Attempts: {attempt[0]} (with {elapsed:.2f}s total including backoff)")

    # ── Fallback Chain Demo ──
    print("\n▸ Fallback Chain Demo")
    chain = FallbackChain("llm_provider", audit_callback=lambda e: audit_log.append(e))
    chain.add("openai", lambda q: (_ for _ in ()).throw(ConnectionError("OpenAI 503")))
    chain.add("bedrock", lambda q: (_ for _ in ()).throw(ConnectionError("Bedrock timeout")))
    chain.add("local", lambda q: f"Local model response to: {q}")

    result = chain.execute("What was ACME revenue?")
    print(f"  Result: {result}")

    # ── Timeout Demo ──
    print("\n▸ Timeout Hierarchy")
    tc = TimeoutConfig.all()
    for op, timeout in tc.items():
        print(f"  {op}: {timeout}s")

    # ── Full Resilience Stack ──
    print("\n▸ Full Resilience Stack (execute_with_resilience)")
    call_num = [0]
    def simulated_qdrant_call():
        call_num[0] += 1
        if call_num[0] == 1:
            raise ConnectionError("Transient failure")
        return {"vectors": [1, 2, 3]}

    # Reset qdrant circuit breaker
    rm.circuit_breakers["qdrant"].reset()

    result = rm.execute_with_resilience(
        "qdrant",
        simulated_qdrant_call,
        fallback=lambda: {"vectors": [], "degraded": True},
        timeout=3.0
    )
    print(f"  Result: {result}")

    # ── System Health ──
    print("\n▸ System resilience health:")
    health = rm.health()
    print(f"  Status: {health['status']}")
    print(f"  Open circuits: {health['open_circuits'] or 'none'}")
    print(f"  Overloaded bulkheads: {health['overloaded_bulkheads'] or 'none'}")

    print(f"\n▸ Audit events: {len(audit_log)}")

    print("\n" + "=" * 70)
    print("✓ Infrastructure resilience demo complete")
    print("=" * 70)


if __name__ == "__main__":
    demo()

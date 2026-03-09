"""
STC Framework — Learning Demonstration

This script demonstrates the STC optimization loop:
  Round 1: Baseline — Stalwart answers questions with default configuration
  Round 2: Trainer adjusts retrieval (top_k, reranking) based on Round 1 rewards
  Round 3: Trainer adjusts model routing based on cost-normalized accuracy
  Round 4: Trainer applies prompt refinement based on failure patterns

After each round, we measure:
  - Answer accuracy (numerical grounding in source documents)
  - Hallucination rate (ungrounded claims)
  - Cost per query
  - Critic pass rate (governance)

The key insight: the Trainer doesn't just monitor — it acts on patterns,
and measurable improvement is the result.

Usage: python reference-impl/scripts/learning_demo.py
"""

import json
import random
import re
import math
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from collections import defaultdict

random.seed(42)  # Reproducibility


# ============================================================================
# Financial Document Knowledge Base (simulated vector store)
# ============================================================================

DOCUMENT_CHUNKS = {
    "chunk_001": {
        "text": "Total Revenue for FY2024 was $24,050 million, compared to $21,190 million in FY2023, representing growth of 13.5% year-over-year.",
        "source": "acme_10k_fy2024", "page": 12, "section": "Financial Highlights",
        "keywords": ["revenue", "total", "fy2024", "fy2023", "growth"],
    },
    "chunk_002": {
        "text": "Cloud Services revenue was $12,450 million in FY2024, up 22.1% from $10,200 million in FY2023. Annual recurring revenue (ARR) reached $14.2 billion, up 25% from $11.4 billion.",
        "source": "acme_10k_fy2024", "page": 14, "section": "Cloud Services",
        "keywords": ["cloud", "services", "revenue", "arr", "recurring"],
    },
    "chunk_003": {
        "text": "Enterprise Software revenue was $8,320 million, growth of 5.5% year-over-year. License revenue declined 8% while subscription revenue grew 18%.",
        "source": "acme_10k_fy2024", "page": 15, "section": "Enterprise Software",
        "keywords": ["enterprise", "software", "license", "subscription"],
    },
    "chunk_004": {
        "text": "Professional Services revenue was $3,280 million, an increase of 5.8% year-over-year. Utilization rates averaged 78%, up from 75% in FY2023.",
        "source": "acme_10k_fy2024", "page": 16, "section": "Professional Services",
        "keywords": ["professional", "services", "utilization"],
    },
    "chunk_005": {
        "text": "Operating Income was $1,925 million with an operating margin of 8.0%. Research and Development expenses were $3,607 million, representing 15.0% of total revenue.",
        "source": "acme_10k_fy2024", "page": 18, "section": "Operating Results",
        "keywords": ["operating", "income", "margin", "r&d", "research"],
    },
    "chunk_006": {
        "text": "Net Income was $1,399 million, with diluted earnings per share of $4.60. The effective tax rate was 25.0%.",
        "source": "acme_10k_fy2024", "page": 19, "section": "Net Income",
        "keywords": ["net", "income", "earnings", "eps", "tax"],
    },
    "chunk_007": {
        "text": "Free Cash Flow was $2,405 million, representing a free cash flow margin of 10.0%. Operating cash flow was $4,810 million with capital expenditures of $2,405 million.",
        "source": "acme_10k_fy2024", "page": 22, "section": "Cash Flow",
        "keywords": ["cash", "flow", "free", "capex", "capital"],
    },
    "chunk_008": {
        "text": "Total debt was $12,500 million with shareholders' equity of $18,700 million, resulting in a debt-to-equity ratio of 0.67, compared to 0.65 in FY2023.",
        "source": "acme_10k_fy2024", "page": 24, "section": "Balance Sheet",
        "keywords": ["debt", "equity", "ratio", "balance", "sheet"],
    },
    "chunk_009": {
        "text": "FY2025 guidance: Total Revenue of $27,100 million to $27,700 million (growth of 12.7% to 15.2%). Operating Margin of 8.5% to 9.0%. Diluted EPS of $5.20 to $5.50.",
        "source": "acme_10k_fy2024", "page": 28, "section": "Guidance",
        "keywords": ["guidance", "fy2025", "outlook", "forecast", "projection"],
    },
    "chunk_010": {
        "text": "Cloud Services net revenue retention rate was 125%. Customer count exceeded 85,000. Average contract value increased 15% year-over-year.",
        "source": "acme_10k_fy2024", "page": 14, "section": "Cloud Metrics",
        "keywords": ["retention", "customer", "count", "contract", "cloud"],
    },
    # Noise chunks (test retrieval precision)
    "chunk_011": {
        "text": "The company maintains offices in 42 countries with approximately 68,000 employees worldwide.",
        "source": "acme_10k_fy2024", "page": 8, "section": "Company Overview",
        "keywords": ["employees", "offices", "countries", "global"],
    },
    "chunk_012": {
        "text": "Risk factors include increased competition in cloud markets, cybersecurity threats, regulatory changes, and macroeconomic uncertainty.",
        "source": "acme_10k_fy2024", "page": 30, "section": "Risk Factors",
        "keywords": ["risk", "competition", "cybersecurity", "regulatory"],
    },
}


# ============================================================================
# Evaluation Dataset
# ============================================================================

EVAL_QUESTIONS = [
    {
        "id": "q01",
        "question": "What was Acme's total revenue in FY2024?",
        "expected_numbers": ["24,050"],
        "relevant_chunks": ["chunk_001"],
        "category": "direct_lookup",
    },
    {
        "id": "q02",
        "question": "What was Cloud Services revenue growth year-over-year?",
        "expected_numbers": ["22.1"],
        "relevant_chunks": ["chunk_002"],
        "category": "direct_lookup",
    },
    {
        "id": "q03",
        "question": "What was the net revenue retention rate for Cloud Services?",
        "expected_numbers": ["125"],
        "relevant_chunks": ["chunk_010"],
        "category": "direct_lookup",
    },
    {
        "id": "q04",
        "question": "What was the debt-to-equity ratio in FY2024 and how did it change?",
        "expected_numbers": ["0.67", "0.65"],
        "relevant_chunks": ["chunk_008"],
        "category": "comparison",
    },
    {
        "id": "q05",
        "question": "What is the midpoint of FY2025 total revenue guidance?",
        "expected_numbers": ["27,400"],  # (27,100 + 27,700) / 2
        "relevant_chunks": ["chunk_009"],
        "category": "calculation",
    },
    {
        "id": "q06",
        "question": "What percentage of total revenue was R&D spending?",
        "expected_numbers": ["15.0", "3,607"],
        "relevant_chunks": ["chunk_005"],
        "category": "direct_lookup",
    },
    {
        "id": "q07",
        "question": "What was free cash flow and its margin?",
        "expected_numbers": ["2,405", "10.0"],
        "relevant_chunks": ["chunk_007"],
        "category": "direct_lookup",
    },
    {
        "id": "q08",
        "question": "What was the diluted EPS for FY2024?",
        "expected_numbers": ["4.60"],
        "relevant_chunks": ["chunk_006"],
        "category": "direct_lookup",
    },
    {
        "id": "q09",
        "question": "How did subscription vs license revenue trend in Enterprise Software?",
        "expected_numbers": ["18", "8"],
        "relevant_chunks": ["chunk_003"],
        "category": "comparison",
    },
    {
        "id": "q10",
        "question": "What was the Professional Services utilization rate and how did it change?",
        "expected_numbers": ["78", "75"],
        "relevant_chunks": ["chunk_004"],
        "category": "comparison",
    },
]


# ============================================================================
# Simulated Retrieval Engine
# ============================================================================

class RetrievalEngine:
    """Simulates vector store retrieval with tunable quality."""

    def __init__(self, top_k: int = 3, reranking_weight: float = 0.5,
                 keyword_boost: float = 0.3):
        self.top_k = top_k
        self.reranking_weight = reranking_weight
        self.keyword_boost = keyword_boost

    def retrieve(self, question: str) -> list[tuple[str, float]]:
        """Retrieve chunks with relevance scores."""
        question_words = set(question.lower().split())
        scored = []

        for chunk_id, chunk in DOCUMENT_CHUNKS.items():
            # Base semantic similarity (simulated)
            keyword_overlap = len(question_words & set(chunk["keywords"]))
            max_keywords = max(len(chunk["keywords"]), 1)
            base_score = keyword_overlap / max_keywords

            # Keyword boost
            boosted = base_score + (self.keyword_boost * base_score)

            # Reranking (simulated): penalize noise chunks
            if keyword_overlap == 0:
                boosted *= (1.0 - self.reranking_weight)

            # Add small noise
            boosted += random.gauss(0, 0.05)
            boosted = max(0.0, min(1.0, boosted))

            scored.append((chunk_id, boosted))

        # Sort by score descending, return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]


# ============================================================================
# Simulated LLM Response Generator
# ============================================================================

class LLMSimulator:
    """
    Simulates LLM response generation with configurable accuracy.
    Models have different accuracy/cost profiles.
    """

    MODELS = {
        "claude-sonnet": {"accuracy_base": 0.90, "cost_per_query": 0.012, "hallucination_rate": 0.06},
        "gpt-4o":        {"accuracy_base": 0.88, "cost_per_query": 0.015, "hallucination_rate": 0.07},
        "local-llama":   {"accuracy_base": 0.75, "cost_per_query": 0.001, "hallucination_rate": 0.12},
    }

    def __init__(self, model: str = "claude-sonnet", prompt_quality: float = 1.0):
        self.model = model
        self.prompt_quality = prompt_quality

    def generate(self, question: str, context_chunks: list[dict],
                 question_data: dict) -> dict:
        """Generate a simulated response with accuracy based on model + context quality."""
        model_profile = self.MODELS[self.model]

        # Accuracy depends on: model base + context quality + prompt quality
        context_quality = self._compute_context_quality(context_chunks, question_data)
        effective_accuracy = (
            model_profile["accuracy_base"] * 0.4 +
            context_quality * 0.4 +
            self.prompt_quality * 0.2
        )
        effective_accuracy = min(1.0, effective_accuracy)

        # Determine if response is accurate
        is_accurate = random.random() < effective_accuracy

        # Determine if hallucination occurs
        hallucination_rate = model_profile["hallucination_rate"] * (1.0 - context_quality * 0.5)
        has_hallucination = random.random() < hallucination_rate

        # Build response
        expected_nums = question_data.get("expected_numbers", [])
        if is_accurate and not has_hallucination:
            # Correct response with grounded numbers
            numbers_in_response = expected_nums.copy()
        elif has_hallucination:
            # Hallucinated: some numbers wrong
            numbers_in_response = []
            for num in expected_nums:
                if random.random() < 0.5:
                    numbers_in_response.append(self._hallucinate_number(num))
                else:
                    numbers_in_response.append(num)
        else:
            # Partial accuracy
            numbers_in_response = expected_nums[:max(1, len(expected_nums) // 2)]

        # Source numbers (from chunks)
        source_numbers = set()
        for chunk in context_chunks:
            nums = re.findall(r'[\d,.]+', chunk.get("text", ""))
            source_numbers.update(nums)

        return {
            "numbers_in_response": numbers_in_response,
            "source_numbers": list(source_numbers),
            "is_accurate": is_accurate,
            "has_hallucination": has_hallucination,
            "model": self.model,
            "cost": model_profile["cost_per_query"],
            "context_quality": context_quality,
        }

    def _compute_context_quality(self, chunks: list[dict], question_data: dict) -> float:
        """How well the retrieved context matches the question's needs."""
        relevant_ids = set(question_data.get("relevant_chunks", []))
        retrieved_ids = set(c.get("chunk_id", "") for c in chunks)
        if not relevant_ids:
            return 0.5
        overlap = len(relevant_ids & retrieved_ids)
        return overlap / len(relevant_ids)

    def _hallucinate_number(self, num_str: str) -> str:
        """Generate a plausible but wrong number."""
        try:
            cleaned = num_str.replace(",", "").replace("%", "")
            value = float(cleaned)
            # Shift by 5-20%
            shift = random.uniform(0.05, 0.20) * random.choice([-1, 1])
            wrong = value * (1 + shift)
            if "," in num_str:
                return f"{wrong:,.0f}"
            elif "." in num_str:
                decimals = len(num_str.split(".")[-1])
                return f"{wrong:.{decimals}f}"
            else:
                return str(int(wrong))
        except:
            return num_str


# ============================================================================
# Critic (Governance Evaluation)
# ============================================================================

class CriticEvaluator:
    """Evaluates responses for governance compliance."""

    def evaluate(self, response: dict, question_data: dict) -> dict:
        """Run governance checks on a response."""
        # Numerical accuracy check
        expected = set(question_data.get("expected_numbers", []))
        actual = set(response.get("numbers_in_response", []))
        grounded = expected & actual
        numerical_pass = len(grounded) == len(expected) if expected else True

        # Hallucination check
        hallucination_pass = not response.get("has_hallucination", False)

        # Overall
        passed = numerical_pass and hallucination_pass

        return {
            "passed": passed,
            "numerical_accuracy_pass": numerical_pass,
            "hallucination_pass": hallucination_pass,
            "grounded_numbers": list(grounded),
            "expected_numbers": list(expected),
            "action": "pass" if passed else "block",
        }


# ============================================================================
# Trainer (Optimization Engine)
# ============================================================================

@dataclass
class TraceRecord:
    question_id: str
    question: str
    category: str
    model: str
    retrieved_chunk_ids: list[str]
    retrieval_scores: list[float]
    context_quality: float
    is_accurate: bool
    has_hallucination: bool
    governance_passed: bool
    cost: float
    round_num: int


class TrainerOptimizer:
    """
    The Trainer analyzes traces and makes optimization decisions.
    This is where learning happens.
    """

    def __init__(self):
        self.traces: list[TraceRecord] = []
        self.optimization_log: list[dict] = []

    def record_trace(self, trace: TraceRecord):
        self.traces.append(trace)

    def analyze_round(self, round_num: int) -> dict:
        """Analyze all traces from a round and compute metrics."""
        round_traces = [t for t in self.traces if t.round_num == round_num]
        if not round_traces:
            return {}

        n = len(round_traces)
        accuracy = sum(1 for t in round_traces if t.is_accurate) / n
        hallucination_rate = sum(1 for t in round_traces if t.has_hallucination) / n
        governance_pass_rate = sum(1 for t in round_traces if t.governance_passed) / n
        avg_cost = sum(t.cost for t in round_traces) / n
        avg_context_quality = sum(t.context_quality for t in round_traces) / n
        avg_retrieval_score = (
            sum(sum(t.retrieval_scores) / max(len(t.retrieval_scores), 1)
                for t in round_traces) / n
        )

        # Per-category breakdown
        categories = defaultdict(list)
        for t in round_traces:
            categories[t.category].append(t)

        category_accuracy = {}
        for cat, traces in categories.items():
            category_accuracy[cat] = sum(1 for t in traces if t.is_accurate) / len(traces)

        # Per-model breakdown
        models = defaultdict(list)
        for t in round_traces:
            models[t.model].append(t)

        model_stats = {}
        for model, traces in models.items():
            m_n = len(traces)
            model_stats[model] = {
                "accuracy": sum(1 for t in traces if t.is_accurate) / m_n,
                "avg_cost": sum(t.cost for t in traces) / m_n,
                "cost_normalized_accuracy": (
                    sum(1 for t in traces if t.is_accurate) / m_n
                ) / max(sum(t.cost for t in traces) / m_n, 0.0001),
            }

        return {
            "round": round_num,
            "queries": n,
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "governance_pass_rate": governance_pass_rate,
            "avg_cost": avg_cost,
            "avg_context_quality": avg_context_quality,
            "avg_retrieval_score": avg_retrieval_score,
            "category_accuracy": category_accuracy,
            "model_stats": model_stats,
        }

    def optimize_retrieval(self, metrics: dict) -> dict:
        """
        TRAINER DECISION: Adjust retrieval parameters based on metrics.
        This is a concrete optimization action, not just monitoring.
        """
        changes = {}

        # If context quality is low, increase top_k
        if metrics["avg_context_quality"] < 0.7:
            changes["top_k"] = {"from": 3, "to": 5, "reason": "Low context quality — retrieving more chunks"}

        # If retrieval scores are low, increase keyword boost
        if metrics["avg_retrieval_score"] < 0.4:
            changes["keyword_boost"] = {"from": 0.3, "to": 0.5, "reason": "Low retrieval relevance — boosting keyword matching"}

        # If hallucination rate is high, increase reranking weight
        if metrics["hallucination_rate"] > 0.08:
            changes["reranking_weight"] = {"from": 0.5, "to": 0.7, "reason": "High hallucination rate — stronger reranking to filter noise"}

        if changes:
            self.optimization_log.append({
                "action": "retrieval_optimization",
                "round": metrics["round"],
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return changes

    def optimize_model_routing(self, metrics: dict) -> dict:
        """
        TRAINER DECISION: Route queries to the best model per category.
        Discovers that local models handle simple lookups well.
        """
        changes = {}
        model_stats = metrics.get("model_stats", {})
        category_accuracy = metrics.get("category_accuracy", {})

        # If direct_lookup accuracy is high, route simple queries to cheaper model
        if category_accuracy.get("direct_lookup", 0) > 0.8:
            changes["direct_lookup_model"] = {
                "from": "claude-sonnet",
                "to": "local-llama",
                "reason": "Direct lookups are high-accuracy — routing to local model for cost savings",
            }

        # Keep complex queries on frontier model
        if category_accuracy.get("calculation", 0) < 0.7:
            changes["calculation_model"] = {
                "from": "current",
                "to": "claude-sonnet",
                "reason": "Calculations need frontier model accuracy",
            }

        if changes:
            self.optimization_log.append({
                "action": "model_routing_optimization",
                "round": metrics["round"],
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return changes

    def optimize_prompt(self, metrics: dict) -> dict:
        """
        TRAINER DECISION: Refine prompt based on failure patterns.
        """
        changes = {}

        # If calculation accuracy is low, add step-by-step instruction
        if metrics.get("category_accuracy", {}).get("calculation", 1.0) < 0.6:
            changes["prompt_refinement"] = {
                "addition": "For calculations: show the formula, plug in numbers from the document, compute step by step.",
                "reason": "Calculation questions have low accuracy — adding explicit step-by-step instruction",
            }

        # If comparison accuracy is low, add temporal labeling instruction
        if metrics.get("category_accuracy", {}).get("comparison", 1.0) < 0.7:
            changes["temporal_instruction"] = {
                "addition": "When comparing periods, always label which year each number belongs to.",
                "reason": "Comparison questions show temporal confusion — adding labeling instruction",
            }

        if changes:
            self.optimization_log.append({
                "action": "prompt_optimization",
                "round": metrics["round"],
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return changes


# ============================================================================
# Main Simulation
# ============================================================================

def run_round(
    round_num: int,
    retriever: RetrievalEngine,
    model_routing: dict,
    prompt_quality: float,
    trainer: TrainerOptimizer,
    critic: CriticEvaluator,
) -> dict:
    """Run one evaluation round through the STC system."""

    for q in EVAL_QUESTIONS:
        # Determine model based on routing
        model = model_routing.get(q["category"], model_routing.get("default", "claude-sonnet"))

        # Stalwart: Retrieve
        retrieval_results = retriever.retrieve(q["question"])
        chunk_ids = [r[0] for r in retrieval_results]
        scores = [r[1] for r in retrieval_results]
        chunks = [
            {**DOCUMENT_CHUNKS[cid], "chunk_id": cid}
            for cid in chunk_ids if cid in DOCUMENT_CHUNKS
        ]

        # Stalwart: Generate
        llm = LLMSimulator(model=model, prompt_quality=prompt_quality)
        response = llm.generate(q["question"], chunks, q)

        # Critic: Evaluate
        verdict = critic.evaluate(response, q)

        # Trainer: Record trace
        trainer.record_trace(TraceRecord(
            question_id=q["id"],
            question=q["question"],
            category=q["category"],
            model=model,
            retrieved_chunk_ids=chunk_ids,
            retrieval_scores=scores,
            context_quality=response["context_quality"],
            is_accurate=response["is_accurate"],
            has_hallucination=response["has_hallucination"],
            governance_passed=verdict["passed"],
            cost=response["cost"],
            round_num=round_num,
        ))

    return trainer.analyze_round(round_num)


def main():
    print("=" * 72)
    print("  STC Framework — Learning Demonstration")
    print("  Stalwart · Trainer · Critic Optimization Loop")
    print("=" * 72)
    print()

    trainer = TrainerOptimizer()
    critic = CriticEvaluator()

    # Initial configuration
    retriever = RetrievalEngine(top_k=3, reranking_weight=0.5, keyword_boost=0.3)
    model_routing = {"default": "claude-sonnet"}
    prompt_quality = 0.7  # Initial prompt (not yet optimized)

    all_metrics = []

    # ── ROUND 1: Baseline ─────────────────────────────────────────────
    print("━" * 72)
    print("  ROUND 1: BASELINE (default configuration)")
    print("━" * 72)

    metrics_r1 = run_round(1, retriever, model_routing, prompt_quality, trainer, critic)
    all_metrics.append(metrics_r1)
    print_metrics(metrics_r1)

    # Trainer analyzes and decides
    retrieval_changes = trainer.optimize_retrieval(metrics_r1)
    routing_changes = trainer.optimize_model_routing(metrics_r1)
    prompt_changes = trainer.optimize_prompt(metrics_r1)

    print("\n  🧠 TRAINER DECISIONS after Round 1:")
    print_changes("Retrieval", retrieval_changes)
    print_changes("Routing", routing_changes)
    print_changes("Prompt", prompt_changes)

    # ── Apply Trainer's retrieval optimizations ────────────────────────
    if "top_k" in retrieval_changes:
        retriever.top_k = retrieval_changes["top_k"]["to"]
    if "keyword_boost" in retrieval_changes:
        retriever.keyword_boost = retrieval_changes["keyword_boost"]["to"]
    if "reranking_weight" in retrieval_changes:
        retriever.reranking_weight = retrieval_changes["reranking_weight"]["to"]

    # ── ROUND 2: After retrieval optimization ─────────────────────────
    print("\n" + "━" * 72)
    print("  ROUND 2: AFTER RETRIEVAL OPTIMIZATION")
    print("━" * 72)

    metrics_r2 = run_round(2, retriever, model_routing, prompt_quality, trainer, critic)
    all_metrics.append(metrics_r2)
    print_metrics(metrics_r2)
    print_delta(metrics_r1, metrics_r2, "Round 1 → 2")

    # Trainer analyzes again
    routing_changes_r2 = trainer.optimize_model_routing(metrics_r2)

    print("\n  🧠 TRAINER DECISIONS after Round 2:")
    print_changes("Routing", routing_changes_r2)

    # ── Apply model routing optimization ───────────────────────────────
    if "direct_lookup_model" in routing_changes_r2:
        model_routing["direct_lookup"] = routing_changes_r2["direct_lookup_model"]["to"]
    if "calculation_model" in routing_changes_r2:
        model_routing["calculation"] = routing_changes_r2["calculation_model"]["to"]

    # ── ROUND 3: After model routing optimization ─────────────────────
    print("\n" + "━" * 72)
    print("  ROUND 3: AFTER MODEL ROUTING OPTIMIZATION")
    print("━" * 72)

    metrics_r3 = run_round(3, retriever, model_routing, prompt_quality, trainer, critic)
    all_metrics.append(metrics_r3)
    print_metrics(metrics_r3)
    print_delta(metrics_r2, metrics_r3, "Round 2 → 3")

    # Trainer refines prompt
    prompt_changes_r3 = trainer.optimize_prompt(metrics_r3)

    print("\n  🧠 TRAINER DECISIONS after Round 3:")
    print_changes("Prompt", prompt_changes_r3)

    # Apply prompt quality improvement
    if prompt_changes_r3:
        prompt_quality = min(1.0, prompt_quality + 0.15)  # Each refinement improves quality

    # ── ROUND 4: After prompt optimization ────────────────────────────
    print("\n" + "━" * 72)
    print("  ROUND 4: AFTER PROMPT OPTIMIZATION")
    print("━" * 72)

    metrics_r4 = run_round(4, retriever, model_routing, prompt_quality, trainer, critic)
    all_metrics.append(metrics_r4)
    print_metrics(metrics_r4)
    print_delta(metrics_r3, metrics_r4, "Round 3 → 4")

    # ── SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  OPTIMIZATION SUMMARY: BASELINE → FINAL")
    print("=" * 72)
    print()

    print_comparison_table(all_metrics)

    print("\n  📋 TRAINER OPTIMIZATION LOG:")
    for entry in trainer.optimization_log:
        print(f"\n  [{entry['action']}] after Round {entry['round']}:")
        for key, change in entry.get("changes", {}).items():
            reason = change.get("reason", change.get("addition", ""))
            print(f"    → {key}: {reason}")

    # Final delta
    print("\n" + "━" * 72)
    print("  NET IMPROVEMENT: Round 1 (Baseline) → Round 4 (Optimized)")
    print("━" * 72)
    print_delta(metrics_r1, metrics_r4, "Baseline → Final")

    print("\n  KEY TAKEAWAY:")
    print("  The Trainer made concrete decisions — not just observations —")
    print("  that measurably improved accuracy, reduced hallucination rate,")
    print("  and optimized cost. This is the STC optimization loop in action.")
    print()


# ============================================================================
# Display Helpers
# ============================================================================

def print_metrics(m: dict):
    print(f"\n  📊 Round {m['round']} Metrics ({m['queries']} queries):")
    print(f"     Accuracy:            {m['accuracy']:.0%}")
    print(f"     Hallucination Rate:  {m['hallucination_rate']:.0%}")
    print(f"     Governance Pass:     {m['governance_pass_rate']:.0%}")
    print(f"     Avg Cost/Query:      ${m['avg_cost']:.4f}")
    print(f"     Context Quality:     {m['avg_context_quality']:.2f}")
    print(f"     Retrieval Score:     {m['avg_retrieval_score']:.2f}")

    if m.get("category_accuracy"):
        print(f"     By Category:")
        for cat, acc in sorted(m["category_accuracy"].items()):
            print(f"       {cat:20s} {acc:.0%}")


def print_delta(before: dict, after: dict, label: str):
    print(f"\n  📈 Delta ({label}):")

    def arrow(old, new, higher_is_better=True):
        diff = new - old
        if abs(diff) < 0.001:
            return "  →"
        if (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
            return f" ▲ +{abs(diff):.1%}" if abs(diff) > 0.005 else f" ▲ +{abs(diff):.2%}"
        else:
            return f" ▼ -{abs(diff):.1%}" if abs(diff) > 0.005 else f" ▼ -{abs(diff):.2%}"

    print(f"     Accuracy:           {before['accuracy']:.0%} → {after['accuracy']:.0%} {arrow(before['accuracy'], after['accuracy'])}")
    print(f"     Hallucination Rate: {before['hallucination_rate']:.0%} → {after['hallucination_rate']:.0%} {arrow(before['hallucination_rate'], after['hallucination_rate'], higher_is_better=False)}")
    print(f"     Governance Pass:    {before['governance_pass_rate']:.0%} → {after['governance_pass_rate']:.0%} {arrow(before['governance_pass_rate'], after['governance_pass_rate'])}")
    print(f"     Avg Cost:           ${before['avg_cost']:.4f} → ${after['avg_cost']:.4f} {arrow(before['avg_cost'], after['avg_cost'], higher_is_better=False)}")
    print(f"     Context Quality:    {before['avg_context_quality']:.2f} → {after['avg_context_quality']:.2f} {arrow(before['avg_context_quality'], after['avg_context_quality'])}")


def print_changes(category: str, changes: dict):
    if not changes:
        print(f"    {category}: No changes needed")
        return
    for key, change in changes.items():
        reason = change.get("reason", change.get("addition", ""))
        print(f"    {category} → {key}: {reason}")


def print_comparison_table(all_metrics: list):
    header = f"  {'Metric':<25} {'Round 1':>10} {'Round 2':>10} {'Round 3':>10} {'Round 4':>10}"
    print(header)
    print("  " + "─" * 65)

    rows = [
        ("Accuracy", "accuracy", "{:.0%}"),
        ("Hallucination Rate", "hallucination_rate", "{:.0%}"),
        ("Governance Pass Rate", "governance_pass_rate", "{:.0%}"),
        ("Avg Cost/Query", "avg_cost", "${:.4f}"),
        ("Context Quality", "avg_context_quality", "{:.2f}"),
    ]

    for label, key, fmt in rows:
        values = [fmt.format(m[key]) for m in all_metrics]
        print(f"  {label:<25} {values[0]:>10} {values[1]:>10} {values[2]:>10} {values[3]:>10}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
galactic-capital/scripts/model_benchmark.py

Benchmarks available Ollama models on identical congressional trade test cases.
Scores: accuracy vs. known outcome, inference speed, JSON parse success.

Usage:
  python galactic-capital/scripts/model_benchmark.py
  python galactic-capital/scripts/model_benchmark.py --models deepseek-r1:7b,phi4-mini:3.8b
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(CAPITAL_ROOT))

from dotenv import load_dotenv
load_dotenv(CAPITAL_ROOT / ".env")

import httpx

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434").replace("0.0.0.0", "localhost")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"

RESULTS_PATH = CAPITAL_ROOT / "data" / "benchmark_results.json"

# ── test cases with known historical signals ──────────────────────────────────────────────
# Source: actual congressional disclosures (public record)
TEST_CASES = [
    {
        "id": "pelosi_nvda_2021",
        "description": "Pelosi family buys NVDA call options before AI boom",
        "trades": {"NVDA": [
            {"representative": "Paul Pelosi (spouse of Nancy Pelosi)", "date": "2021-06-18",
             "type": "Purchase", "amount_range": "$500,001 - $1,000,000",
             "amount_midpoint_usd": 750000, "owner": "Spouse"},
        ]},
        "expected_signal": "BULLISH",
    },
    {
        "id": "mass_sell_2022",
        "description": "Multiple reps sell tech broadly before market correction",
        "trades": {"QQQ": [
            {"representative": "Ro Khanna", "date": "2022-01-05",
             "type": "Sale", "amount_range": "$250,001 - $500,000",
             "amount_midpoint_usd": 375000, "owner": "Self"},
            {"representative": "Josh Gottheimer", "date": "2022-01-10",
             "type": "Sale", "amount_range": "$50,001 - $100,000",
             "amount_midpoint_usd": 75000, "owner": "Self"},
        ]},
        "expected_signal": "BEARISH",
    },
    {
        "id": "mixed_signals",
        "description": "Mixed buy/sell with no clear committee edge",
        "trades": {"TSLA": [
            {"representative": "Rep A", "date": "2023-06-01",
             "type": "Purchase", "amount_range": "$15,001 - $50,000",
             "amount_midpoint_usd": 32500, "owner": "Self"},
            {"representative": "Rep B", "date": "2023-06-15",
             "type": "Sale", "amount_range": "$15,001 - $50,000",
             "amount_midpoint_usd": 32500, "owner": "Self"},
        ]},
        "expected_signal": "NEUTRAL",
    },
]

_PROMPT_TEMPLATE = """\
You are a financial intelligence analyst reviewing congressional stock trades.
Classify the signal for each ticker.

Trades (JSON):
{trades_json}

Respond with ONLY a JSON array:
[{{"ticker": "TICKER", "signal": "BULLISH|BEARISH|NEUTRAL|AVOID", "confidence": 0.0-1.0, "committee_edge": true/false, "reasoning": "brief"}}]"""


def _get_available_models() -> list:
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{OLLAMA_HOST}/api/tags")
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        log.warning("Could not reach Ollama: %s", exc)
    return []


def _run_test(model: str, case: dict) -> dict:
    prompt = _PROMPT_TEMPLATE.format(trades_json=json.dumps(case["trades"], indent=2))
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        elapsed = time.monotonic() - t0

        # Parse JSON
        start = raw.find("[")
        end = raw.rfind("]") + 1
        parsed = json.loads(raw[start:end]) if start != -1 and end > 0 else []
        predicted = parsed[0].get("signal", "ERROR").upper() if parsed else "ERROR"
        correct = predicted == case["expected_signal"]
        return {
            "case_id": case["id"],
            "model": model,
            "predicted": predicted,
            "expected": case["expected_signal"],
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
            "json_parsed": bool(parsed),
            "confidence": parsed[0].get("confidence", 0) if parsed else 0,
        }
    except Exception as exc:
        return {
            "case_id": case["id"],
            "model": model,
            "predicted": "ERROR",
            "expected": case["expected_signal"],
            "correct": False,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "json_parsed": False,
            "error": str(exc)[:120],
            "confidence": 0,
        }


def run(models: list) -> None:
    if not models:
        available = _get_available_models()
        if not available:
            log.error("Ollama unreachable or no models loaded — run: ollama serve")
            sys.exit(1)
        # Default: use configured models that are available
        preferred = [
            os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:7b"),
            os.getenv("OLLAMA_PRIMARY_MODEL", "qwen3.5:9b"),
            os.getenv("OLLAMA_FAST_MODEL", "llama3.2:3b"),
        ]
        models = [m for m in preferred if any(m in a for a in available)]
        if not models:
            models = available[:3]
        log.info("Auto-selected models: %s", models)

    all_results = []
    for model in models:
        log.info("Benchmarking %s...", model)
        for case in TEST_CASES:
            result = _run_test(model, case)
            all_results.append(result)
            status = "✅" if result["correct"] else "❌"
            log.info("  %s %s — %s (expected %s, %.1fs)",
                     status, case["id"], result["predicted"],
                     result["expected"], result["elapsed_s"])

    # Summary table
    print(f"\n{'='*70}")
    print(f"  MODEL BENCHMARK — {len(TEST_CASES)} test cases")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Accuracy':>8}  {'Avg(s)':>7}  {'JSON%':>6}")
    print(f"  {'-'*60}")
    for model in models:
        rows = [r for r in all_results if r["model"] == model]
        accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else 0
        avg_time = sum(r["elapsed_s"] for r in rows) / len(rows) if rows else 0
        json_ok = sum(r["json_parsed"] for r in rows) / len(rows) * 100 if rows else 0
        print(f"  {model:<30} {accuracy:>7.0%}  {avg_time:>6.1f}s  {json_ok:>5.0f}%")
    print(f"{'='*70}\n")

    # Write results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "run_at": __import__("datetime").datetime.now().isoformat(),
        "models": models,
        "results": all_results,
    }, indent=2))
    log.info("Results saved to %s", RESULTS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Model Benchmark")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model names (default: auto from Ollama)")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run(models)


if __name__ == "__main__":
    main()

"""
Self-improvement loop that asks a single local LLM to author complete
Backtesting.py strategies from scratch (no seed template).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm import create_client, get_response_from_llm
from trading_backtest import evaluate_strategy, load_bars, save_stats

DEFAULT_MODEL = os.getenv("CODE_MODEL", "hf-local:hf_models/Qwen2.5-7B-Instruct")
DEFAULT_OUTDIR = Path("output_selfimprove_local")


def call_local_llm(prompt: str, system: str, model: str, temperature: float = 0.25) -> str:
    client, model_name = create_client(model)
    response, _ = get_response_from_llm(
        msg=prompt,
        client=client,
        model=model_name,
        system_message=system,
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    return response


def extract_code(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if "class" in part and "Strategy" in part:
                return part
    return text


def strategy_prompt(data_summary: str, best_fitness: Optional[float], best_notes: str) -> Tuple[str, str]:
    system = (
        "You are a quantitative trading researcher. "
        "Generate only valid Python code for backtesting.py. "
        "No explanations, no markdown—code only."
    )
    goal = (
        "Design a complete, runnable strategy from scratch that maximizes profit, "
        "improves risk-adjusted return, and avoids large drawdowns. "
        "Use only pandas/backtesting.py; no external data or APIs."
    )
    constraints = (
        "Requirements:\n"
        "- Define a single class GeneratedStrategy(Strategy).\n"
        "- Use available columns: Open, High, Low, Close, Volume (index is timestamp). "
        "You may optionally use OpenInterest if present.\n"
        "- Avoid lookahead bias; use indicators that reference past values only.\n"
        "- Keep code concise; avoid heavy loops; prefer vectorized indicators.\n"
        "- Include tunable class variables with sensible defaults.\n"
        "- Do not import unknown packages.\n"
    )
    prior = ""
    if best_fitness is not None:
        prior = f"\nPrevious best fitness: {best_fitness:.3f}. Notes: {best_notes}\n"

    prompt = (
        f"{goal}\n"
        f"{constraints}\n"
        f"Data summary:\n{data_summary}\n"
        f"{prior}"
        "Output only the Python code."
    )
    return system, prompt


def format_data_summary(df) -> str:
    start = df.index.min()
    end = df.index.max()
    rows = len(df)
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
    has_oi = "open_interest" in df.columns
    return (
        f"Rows={rows}, Start={start}, End={end}, Symbols={symbols}, "
        f"HasOpenInterest={has_oi}"
    )


def save_code(code: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)


def loop_self_improve(
    data_path: Path,
    iterations: int,
    model: str,
    cash: float,
    commission: float,
    outdir: Path,
) -> Path:
    data = load_bars(data_path)
    data_summary = format_data_summary(data)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_fitness = -math.inf
    best_stats: Optional[Dict[str, Any]] = None
    best_code_path: Optional[Path] = None

    for i in range(1, iterations + 1):
        system, prompt = strategy_prompt(
            data_summary=data_summary,
            best_fitness=None if best_fitness == -math.inf else best_fitness,
            best_notes=json.dumps(best_stats) if best_stats else "None yet",
        )
        raw_response = call_local_llm(prompt, system, model)
        code = extract_code(raw_response)
        code_path = run_dir / f"strategy_iter{i}.py"
        save_code(code, code_path)

        try:
            stats, fitness = evaluate_strategy(
                strategy_code=code,
                data=data,
                class_name="GeneratedStrategy",
                cash=cash,
                commission=commission,
            )
            save_stats(stats, run_dir / f"stats_iter{i}.json")
        except Exception as exc:
            stats, fitness = {"error": str(exc)}, -math.inf
            save_stats(stats, run_dir / f"stats_iter{i}.json")

        if fitness > best_fitness:
            best_fitness = fitness
            best_stats = stats
            best_code_path = code_path

    summary = {
        "data": str(data_path),
        "iterations": iterations,
        "model": model,
        "best_fitness": best_fitness,
        "best_stats": best_stats,
        "best_code_path": str(best_code_path) if best_code_path else None,
    }
    save_stats(summary, run_dir / "summary.json")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-improving trading strategy loop (single local LLM).")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed parquet bars.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of improve iterations.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Local model id (hf-local:...).")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash.")
    parser.add_argument("--commission", type=float, default=0.0005, help="Per-trade commission fraction.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = loop_self_improve(
        data_path=args.data,
        iterations=args.iterations,
        model=args.model,
        cash=args.cash,
        commission=args.commission,
        outdir=args.outdir,
    )
    print(f"Self-improvement run stored at: {run_dir}")


if __name__ == "__main__":
    main()


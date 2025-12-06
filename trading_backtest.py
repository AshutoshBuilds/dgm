"""
Utility to run backtests on resampled bar data using backtesting.py.
The LLM is expected to emit a complete Strategy subclass.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from backtesting import Backtest
from backtesting.lib import Strategy


def load_bars(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def evaluate_strategy(
    strategy_code: str,
    data: pd.DataFrame,
    class_name: str = "GeneratedStrategy",
    cash: float = 100_000.0,
    commission: float = 0.0005,
    exclusive_orders: bool = True,
) -> Tuple[Dict[str, Any], float]:
    local_ns: Dict[str, Any] = {}
    # Expose backtesting utilities to strategy code
    local_ns.update({
        "Strategy": Strategy,
        "pd": pd,
    })
    exec(strategy_code, local_ns)
    if class_name not in local_ns:
        raise ValueError(f"Strategy class '{class_name}' not found in generated code.")
    StrategyCls = local_ns[class_name]

    bt_data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    bt = Backtest(
        bt_data,
        StrategyCls,
        cash=cash,
        commission=commission,
        exclusive_orders=exclusive_orders,
    )
    stats = bt.run()
    # Fitness: reward return, penalize drawdown
    ann_return = float(stats.get("Return [%]", 0.0))
    drawdown = float(stats.get("Max. Drawdown [%]", 0.0))
    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    fitness = ann_return - max(drawdown, 0) * 0.5 + sharpe * 5
    return stats, fitness


def save_stats(stats: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)



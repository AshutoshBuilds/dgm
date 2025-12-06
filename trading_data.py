"""
Tick ingestion and resampling for local Indian futures/options data.

Expected CSV columns (case-insensitive):
Ticker, Date, Time, LTP, BuyPrice, BuyQty, SellPrice, SellQty, LTQ, OpenInterest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_TZ = "Asia/Kolkata"
REQUIRED_COLS = [
    "ticker",
    "date",
    "time",
    "ltp",
    "buyprice",
    "buyqty",
    "sellprice",
    "sellqty",
    "ltq",
    "openinterest",
]


@dataclass
class ResampleConfig:
    freq: str = "1min"
    tz: str = DEFAULT_TZ
    output_dir: Path = Path("data/processed")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=renamed)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def load_tick_csv(path: Path, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    if tz:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    def _instrument_type(ticker: str) -> str:
        t = str(ticker).upper()
        if "CE" in t or "PE" in t:
            return "OPTION"
        return "FUTURE"

    df["instrument_type"] = df["ticker"].astype(str).apply(_instrument_type)
    df["symbol"] = df["ticker"].astype(str).str.split(".", n=1).str[0]

    cols = [
        "timestamp",
        "symbol",
        "instrument_type",
        "ltp",
        "buyprice",
        "buyqty",
        "sellprice",
        "sellqty",
        "ltq",
        "openinterest",
    ]
    return df[cols]


def resample_ticks(df: pd.DataFrame, cfg: ResampleConfig, symbol: Optional[str] = None) -> pd.DataFrame:
    if symbol:
        df = df[df["symbol"] == symbol]
    if df.empty:
        raise ValueError("No rows available after filtering")

    df = df.set_index("timestamp")
    grouped = df.groupby("symbol")

    def _agg(group: pd.DataFrame) -> pd.DataFrame:
        agg = group.resample(cfg.freq).agg(
            open=("ltp", "first"),
            high=("ltp", "max"),
            low=("ltp", "min"),
            close=("ltp", "last"),
            volume=("ltq", "sum"),
            open_interest=("openinterest", "last"),
        )
        agg = agg.dropna(subset=["open", "high", "low", "close"])
        agg["symbol"] = group["symbol"].iloc[0]
        agg["instrument_type"] = group["instrument_type"].iloc[0]
        return agg.reset_index()

    resampled = pd.concat([_agg(g) for _, g in grouped], ignore_index=True)
    resampled = resampled.sort_values(["symbol", "timestamp"])
    return resampled


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def discover_csvs(raw_root: Path) -> Iterable[Path]:
    return raw_root.rglob("*.csv")


def run_resample(raw_root: Path, cfg: ResampleConfig, symbols: Optional[set[str]]) -> list[Path]:
    outputs = []
    for csv_path in discover_csvs(raw_root):
        try:
            df = load_tick_csv(csv_path, tz=cfg.tz)
            if symbols:
                df = df[df["symbol"].isin(symbols)]
                if df.empty:
                    continue
            resampled = resample_ticks(df, cfg)
            for sym, g in resampled.groupby("symbol"):
                fname = f"{sym}_{cfg.freq}.parquet"
                out_path = cfg.output_dir / fname
                write_parquet(g, out_path)
                outputs.append(out_path)
        except Exception as exc:  # best-effort
            print(f"[warn] failed {csv_path}: {exc}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest tick CSVs and resample to OHLCV bars.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Root folder containing tick CSVs.")
    parser.add_argument("--freq", type=str, default="1min", help="Bar frequency (e.g., 1min, 5min).")
    parser.add_argument("--tz", type=str, default=DEFAULT_TZ, help="Timezone to localize timestamps.")
    parser.add_argument("--symbols", type=str, nargs="*", help="Optional symbol filters (e.g., NIFTY-I.NFO).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory for bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ResampleConfig(freq=args.freq, tz=args.tz, output_dir=args.output_dir)
    symbols = set(args.symbols) if args.symbols else None
    outputs = run_resample(args.raw_root, cfg, symbols)
    if outputs:
        print(f"Wrote {len(outputs)} parquet file(s).")
    else:
        print("No parquet files written; check input filters.")


if __name__ == "__main__":
    main()


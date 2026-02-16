#!/usr/bin/env python3
"""
Booster box price charts and expected value report for Sorcery TCG.

Shows a combined view per set: box price history, EV history, and current
profitability metrics — all in the terminal.

Usage:
  python booster_box_report.py
  python booster_box_report.py --height 20
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from typing import Optional

import asciichartpy
import numpy as np
import pandas as pd

from colors import (
    header, subheader, bold, dim, cyan, yellow, green, red,
    bright_cyan, bright_green, bright_yellow, price_gain, price_loss, price_change,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Timestamp patterns for CSV filenames
PRICE_RE = re.compile(r"^(\d{8}_\d{4})_sorcery_prices\.csv$")
EV_RE = re.compile(r"^(\d{8}_\d{4})_ev_table\.csv$")


# ── Data loading ────────────────────────────────────────────────────


def discover_files(pattern: re.Pattern) -> list[tuple[str, str]]:
    """Find CSVs matching a timestamp pattern, return [(label, filepath)] sorted by time."""
    pairs = []
    for fname in os.listdir(DATA_DIR):
        m = pattern.match(fname)
        if m:
            ts_raw = m.group(1)
            label = datetime.strptime(ts_raw, "%Y%m%d_%H%M").strftime("%m/%d %H:%M")
            pairs.append((label, os.path.join(DATA_DIR, fname)))
    pairs.sort()
    return pairs


def load_box_prices(snapshots: list[tuple[str, str]]) -> dict[str, list[Optional[float]]]:
    """Load booster box prices per set across all price snapshots.

    Returns {set_name: [price_or_None_per_snapshot, ...]}.
    """
    box_re = re.compile(r"booster.*box", re.I)
    case_re = re.compile(r"\bcase\b", re.I)
    sets: dict[str, list[Optional[float]]] = {}

    for _label, path in snapshots:
        df = pd.read_csv(path)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        # Find booster box rows (standard finish), excluding cases
        boxes = df[
            df["name"].str.contains(box_re, regex=True, na=False)
            & ~df["name"].str.contains(case_re, regex=True, na=False)
            & (df["finish"].str.lower().str.strip() == "standard")
            & df["price"].notna()
        ]
        # Collect prices per expansion
        snap_prices: dict[str, float] = {}
        for _, row in boxes.iterrows():
            exp = str(row["expansion"]).strip()
            price = float(row["price"])
            # Keep highest if multiple listings
            if exp not in snap_prices or price > snap_prices[exp]:
                snap_prices[exp] = price

        # Record all known sets
        for exp in snap_prices:
            if exp not in sets:
                sets[exp] = [None] * len(sets.get(exp, []))

        # Pad and append
        for exp in sets:
            sets[exp].append(snap_prices.get(exp))

    return sets


def load_ev_history(ev_files: list[tuple[str, str]]) -> dict[str, list[Optional[float]]]:
    """Load box EV per set across all EV table snapshots.

    Returns {set_name: [ev_or_None_per_snapshot, ...]}.
    """
    sets: dict[str, list[Optional[float]]] = {}

    for _label, path in ev_files:
        df = pd.read_csv(path)
        snap_evs: dict[str, float] = {}
        for _, row in df.iterrows():
            exp = str(row["Set"]).strip()
            box_ev = row.get("Box EV ($)")
            box_price = row.get("Box Price ($)")
            if pd.notna(box_ev) and pd.notna(box_price):
                snap_evs[exp] = float(box_ev)

        for exp in snap_evs:
            if exp not in sets:
                sets[exp] = [None] * len(sets.get(exp, []))

        for exp in sets:
            sets[exp].append(snap_evs.get(exp))

    return sets


def align_series(
    price_labels: list[str],
    ev_labels: list[str],
    price_data: list[Optional[float]],
    ev_data: list[Optional[float]],
) -> tuple[list[str], list[Optional[float]], list[Optional[float]]]:
    """Align price and EV series onto a merged timeline by label."""
    all_labels = sorted(set(price_labels) | set(ev_labels))

    price_map = dict(zip(price_labels, price_data))
    ev_map = dict(zip(ev_labels, ev_data))

    aligned_prices = [price_map.get(l) for l in all_labels]
    aligned_evs = [ev_map.get(l) for l in all_labels]

    return all_labels, aligned_prices, aligned_evs


# ── Chart rendering ─────────────────────────────────────────────────


def make_chart(
    labels: list[str],
    price_series: list[Optional[float]],
    ev_series: list[Optional[float]],
    height: int = 15,
) -> str:
    """Render a dual-series ASCII chart (box price + EV)."""
    clean_prices = [p if (p is not None and pd.notna(p)) else float("nan") for p in price_series]
    clean_evs = [p if (p is not None and pd.notna(p)) else float("nan") for p in ev_series]

    series = []
    legend_parts = []

    has_prices = any(pd.notna(v) for v in clean_prices)
    has_evs = any(pd.notna(v) for v in clean_evs)

    if has_prices:
        series.append(clean_prices)
        legend_parts.append(cyan("━ Box Price"))
    if has_evs:
        series.append(clean_evs)
        legend_parts.append(green("━ Box EV"))

    if not series:
        return "  (no data)"

    config = {"height": height, "format": "${:,.0f}"}

    if len(series) == 1:
        chart_str = asciichartpy.plot(series[0], config)
    else:
        chart_str = asciichartpy.plot(series, config)

    # X-axis labels
    n = len(labels)
    if n <= 1:
        indices = list(range(n))
    else:
        step = max(1, (n - 1) // min(5, n - 1))
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)

    label_parts = [f"[{idx}]={labels[idx]}" for idx in indices]
    x_axis = "  " + "  ".join(label_parts)

    legend = "  Legend: " + "  |  ".join(legend_parts)
    return chart_str + "\n" + x_axis + "\n" + legend


def format_ev_ratio(ratio: float) -> str:
    """Color-code the EV/Price ratio."""
    text = f"{ratio:.2f}x"
    if ratio >= 1.5:
        return bright_green(bold(text))
    elif ratio >= 1.0:
        return green(text)
    elif ratio >= 0.8:
        return yellow(text)
    else:
        return red(text)


def print_box_report(
    set_name: str,
    labels: list[str],
    price_series: list[Optional[float]],
    ev_series: list[Optional[float]],
    height: int = 15,
) -> None:
    """Print the full report for one booster box."""
    # Current values (last non-None)
    curr_price = next((v for v in reversed(price_series) if v is not None and pd.notna(v)), None)
    curr_ev = next((v for v in reversed(ev_series) if v is not None and pd.notna(v)), None)

    # Price trend
    valid_prices = [v for v in price_series if v is not None and pd.notna(v)]
    price_trend = ""
    if len(valid_prices) >= 2:
        if valid_prices[-1] > valid_prices[0]:
            price_trend = price_gain(" ▲")
        elif valid_prices[-1] < valid_prices[0]:
            price_trend = price_loss(" ▼")

    print(f"\n{header('=' * 64)}")
    print(f"  {bold(set_name)} Booster Box")
    print(f"{header('=' * 64)}")

    # Stats line
    stats = []
    if curr_price is not None:
        stats.append(f"Box Price: {bold(f'${curr_price:,.2f}')}")
    if curr_ev is not None:
        stats.append(f"Box EV: {bold(f'${curr_ev:,.2f}')}")
    if curr_price is not None and curr_ev is not None:
        diff = curr_ev - curr_price
        diff_str = f"${diff:+,.2f}"
        stats.append(f"EV - Price: {price_change(diff, diff_str)}")
        if curr_price > 0:
            ratio = curr_ev / curr_price
            stats.append(f"Ratio: {format_ev_ratio(ratio)}")
    if stats:
        print(f"  {' | '.join(stats)}{price_trend}")

    # Chart
    chart = make_chart(labels, price_series, ev_series, height=height)
    print()
    print(chart)


def print_summary_table(
    price_labels: list[str],
    ev_labels: list[str],
    box_prices: dict[str, list[Optional[float]]],
    box_evs: dict[str, list[Optional[float]]],
) -> None:
    """Print a summary table comparing all sets."""
    print(f"\n{header('=' * 74)}")
    print(header("  BOOSTER BOX SUMMARY"))
    print(f"{header('=' * 74)}")

    rows = []
    for set_name in sorted(set(box_prices.keys()) & set(box_evs.keys())):
        prices = box_prices.get(set_name, [])
        evs = box_evs.get(set_name, [])

        curr_price = next((v for v in reversed(prices) if v is not None and pd.notna(v)), None)
        curr_ev = next((v for v in reversed(evs) if v is not None and pd.notna(v)), None)

        if curr_price is None or curr_ev is None:
            continue

        diff = curr_ev - curr_price
        ratio = curr_ev / curr_price if curr_price > 0 else 0

        rows.append((set_name, curr_price, curr_ev, diff, ratio))

    # Sort by ratio descending
    rows.sort(key=lambda r: r[4], reverse=True)

    # Header
    print(f"\n  {'Set':<25} {'Box Price':>10} {'Box EV':>10} {'EV-Price':>10} {'Ratio':>8}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")

    for set_name, price, ev, diff, ratio in rows:
        diff_str = f"${diff:+,.2f}"
        colored_diff = price_change(diff, diff_str)
        colored_ratio = format_ev_ratio(ratio)
        verdict = price_gain(" BUY") if ratio >= 1.0 else price_loss(" PASS")
        print(
            f"  {set_name:<25} {f'${price:,.2f}':>10} {f'${ev:,.2f}':>10} "
            f"{colored_diff:>20} {colored_ratio:>16}{verdict}"
        )

    print()
    print(f"  {dim('Ratio > 1.0 = positive expected value (EV exceeds box price)')}")
    print(f"  {dim('EV is based on community odds: 36 packs/box, 1/5 unique rate, 1/4 foil rate')}")
    print(f"{header('=' * 74)}")


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Booster box price charts and expected value report."
    )
    p.add_argument("--height", type=int, default=15, help="Chart height in rows (default: 15)")
    args = p.parse_args(argv)

    # Discover data files
    price_snapshots = discover_files(PRICE_RE)
    ev_snapshots = discover_files(EV_RE)

    if not price_snapshots:
        print("No sorcery_prices CSV files found in data/", file=sys.stderr)
        return 1
    if not ev_snapshots:
        print("No ev_table CSV files found in data/", file=sys.stderr)
        return 1

    price_labels = [label for label, _ in price_snapshots]
    ev_labels = [label for label, _ in ev_snapshots]

    print(f"Loaded {bold(str(len(price_snapshots)))} price snapshots "
          f"and {bold(str(len(ev_snapshots)))} EV snapshots")
    print(f"Spanning {bright_cyan(price_labels[0])} to {bright_cyan(price_labels[-1])}")

    # Load data
    box_prices = load_box_prices(price_snapshots)
    box_evs = load_ev_history(ev_snapshots)

    # Sets that have both box price and EV data
    reportable = sorted(set(box_prices.keys()) & set(box_evs.keys()))

    if not reportable:
        print("No sets found with both box prices and EV data.")
        return 1

    print(f"Found {bold(str(len(reportable)))} sets with booster box data: "
          f"{', '.join(reportable)}")

    # Per-set reports
    for set_name in reportable:
        labels, aligned_prices, aligned_evs = align_series(
            price_labels, ev_labels,
            box_prices[set_name], box_evs[set_name],
        )
        print_box_report(set_name, labels, aligned_prices, aligned_evs, height=args.height)

    # Summary table
    print_summary_table(price_labels, ev_labels, box_prices, box_evs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

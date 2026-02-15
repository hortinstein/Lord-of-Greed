#!/usr/bin/env python3
"""
ASCII price charts for Sorcery TCG products.

Uses asciichartpy to render price history from timestamped CSV snapshots.

Usage:
  # Chart all sealed products (booster boxes, packs, cases, etc.)
  python chart_prices.py sealed

  # Chart all versions of a specific card
  python chart_prices.py card "Abundance"

  # Case-insensitive partial match
  python chart_prices.py card "headless haunt"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from typing import Optional

import asciichartpy
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

SEALED_NAME_RE = re.compile(
    r"\b(?:booster|box|case|pack|display|starter|precon|deck)\b", re.I
)

# Timestamp pattern at start of CSV filenames: YYYYMMDD_HHMM
TS_RE = re.compile(r"^(\d{8}_\d{4})_sorcery_prices\.csv$")


def discover_snapshots() -> list[tuple[str, str]]:
    """Find all sorcery_prices CSVs and return [(timestamp_label, filepath)] sorted by time."""
    pairs = []
    for fname in os.listdir(DATA_DIR):
        m = TS_RE.match(fname)
        if m:
            ts_raw = m.group(1)
            label = datetime.strptime(ts_raw, "%Y%m%d_%H%M").strftime("%m/%d %H:%M")
            pairs.append((label, os.path.join(DATA_DIR, fname)))
    pairs.sort()
    return pairs


def load_all_snapshots() -> tuple[list[str], pd.DataFrame]:
    """Load every snapshot CSV and tag rows with the snapshot timestamp.

    Returns (list_of_labels, combined_dataframe_with_'snapshot'_column).
    """
    snapshots = discover_snapshots()
    if not snapshots:
        print("No sorcery_prices CSV files found in data/", file=sys.stderr)
        sys.exit(1)

    frames = []
    labels = []
    for label, path in snapshots:
        df = pd.read_csv(path)
        df["snapshot"] = label
        frames.append(df)
        labels.append(label)

    combined = pd.concat(frames, ignore_index=True)
    combined["price"] = pd.to_numeric(combined["price"], errors="coerce")
    return labels, combined


def is_sealed(name: str) -> bool:
    return bool(SEALED_NAME_RE.search(name))


def chart_series(title: str, labels: list[str], prices: list[Optional[float]],
                 height: int = 15) -> None:
    """Print a single ASCII chart with a title and x-axis labels."""
    # Replace None / NaN with asciichartpy's sentinel for gaps
    clean = [p if (p is not None and pd.notna(p)) else float("nan") for p in prices]

    # Skip if all NaN
    if all(pd.isna(v) for v in clean):
        return

    valid = [v for v in clean if pd.notna(v)]
    lo, hi = min(valid), max(valid)
    price_range = f"${lo:.2f}" if lo == hi else f"${lo:.2f} – ${hi:.2f}"

    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"  Range: {price_range}   Points: {len(valid)}")
    print(f"{'─' * 60}")

    chart = asciichartpy.plot(clean, {"height": height, "format": "${:,.2f}"})
    print(chart)

    # X-axis labels (show first, last, and a few in between)
    n = len(labels)
    if n <= 1:
        indices = list(range(n))
    else:
        step = max(1, (n - 1) // min(5, n - 1))
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)

    # Build a simple label bar
    label_parts = []
    for idx in indices:
        label_parts.append(f"[{idx}]={labels[idx]}")
    print("  " + "  ".join(label_parts))


# ── Sealed products ─────────────────────────────────────────────────


def chart_sealed(labels: list[str], df: pd.DataFrame, height: int = 15) -> None:
    """Chart price history for every sealed product across snapshots."""
    sealed_df = df[df["name"].apply(is_sealed)].copy()
    if sealed_df.empty:
        print("No sealed products found in data.")
        return

    # Build a unique key per product: (name, expansion, finish)
    sealed_df["key"] = sealed_df.apply(
        lambda r: (r["name"], r["expansion"], r["finish"]), axis=1
    )
    unique_keys = sorted(sealed_df["key"].unique(), key=lambda k: (k[1], k[0], k[2]))

    print(f"\n{'=' * 60}")
    print(f"  SEALED PRODUCT PRICE HISTORY  ({len(unique_keys)} products)")
    print(f"{'=' * 60}")

    for key in unique_keys:
        name, expansion, finish = key
        subset = sealed_df[
            (sealed_df["name"] == name)
            & (sealed_df["expansion"] == expansion)
            & (sealed_df["finish"] == finish)
        ]
        # Build price series aligned to snapshot labels
        snap_prices = dict(zip(subset["snapshot"], subset["price"]))
        prices = [snap_prices.get(lbl) for lbl in labels]

        finish_tag = " [Foil]" if str(finish).lower() == "foil" else ""
        title = f"{name}{finish_tag}  ({expansion})"
        chart_series(title, labels, prices, height=height)


# ── Card lookup ──────────────────────────────────────────────────────


def chart_card(card_query: str, labels: list[str], df: pd.DataFrame,
               height: int = 15) -> None:
    """Chart price history for all versions of a card matching the query."""
    pattern = re.compile(re.escape(card_query), re.I)
    # Exclude sealed products from card search
    cards = df[~df["name"].apply(is_sealed)].copy()
    matches = cards[cards["name"].str.contains(pattern, regex=True, na=False)]

    if matches.empty:
        print(f"No cards found matching '{card_query}'.")
        # Suggest close matches
        all_names = cards["name"].dropna().unique()
        suggestions = [n for n in all_names if card_query.lower() in n.lower()]
        if suggestions:
            print("Did you mean one of these?")
            for s in sorted(set(suggestions))[:10]:
                print(f"  - {s}")
        return

    matches["key"] = matches.apply(
        lambda r: (r["name"], r["expansion"], r["finish"]), axis=1
    )
    unique_keys = sorted(matches["key"].unique(), key=lambda k: (k[0], k[1], k[2]))

    print(f"\n{'=' * 60}")
    print(f"  PRICE HISTORY FOR: '{card_query}'  ({len(unique_keys)} versions)")
    print(f"{'=' * 60}")

    for key in unique_keys:
        name, expansion, finish = key
        subset = matches[
            (matches["name"] == name)
            & (matches["expansion"] == expansion)
            & (matches["finish"] == finish)
        ]
        snap_prices = dict(zip(subset["snapshot"], subset["price"]))
        prices = [snap_prices.get(lbl) for lbl in labels]

        finish_tag = " [Foil]" if str(finish).lower() == "foil" else ""
        rarity = subset["rarity"].iloc[0] if "rarity" in subset.columns else ""
        rarity_tag = f"  {rarity}" if rarity and str(rarity) != "nan" else ""
        title = f"{name}{finish_tag}  ({expansion}){rarity_tag}"
        chart_series(title, labels, prices, height=height)


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="ASCII price charts for Sorcery TCG products."
    )
    sub = p.add_subparsers(dest="command")

    sealed_p = sub.add_parser("sealed", help="Chart all sealed products over time")
    sealed_p.add_argument("--height", type=int, default=15, help="Chart height (rows)")

    card_p = sub.add_parser("card", help="Chart all versions of a specific card")
    card_p.add_argument("query", help="Card name (case-insensitive partial match)")
    card_p.add_argument("--height", type=int, default=15, help="Chart height (rows)")

    args = p.parse_args(argv)
    if not args.command:
        p.print_help()
        return 1

    labels, df = load_all_snapshots()
    print(f"Loaded {len(discover_snapshots())} snapshots spanning {labels[0]} to {labels[-1]}")

    if args.command == "sealed":
        chart_sealed(labels, df, height=args.height)
    elif args.command == "card":
        chart_card(args.query, labels, df, height=args.height)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

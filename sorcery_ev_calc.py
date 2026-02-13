#!/usr/bin/env python3
"""
Sorcery EV calculator (community-odds rough model)

Inputs:
  - CSV like your sorcery_prices.csv with (at least) columns:
      name, price, expansion, finish, rarity, art_link

Outputs:
  - Prints an EV table by set (expansion) including sealed pack/box/case prices found in the same CSV
  - Shows biggest price changes since last run (if a previous CSV is found)
  - Optionally writes the table to a CSV

Notes:
  - This uses configurable community odds (defaults are common rough estimates):
      * Pack structure: 11 Ordinary, 3 Exceptional, 1 "rare slot" (Elite or Unique)
      * Unique rate in rare slot: 1/5 packs
      * Foil rate: 1/4 packs
      * Foil rarity mix: Ordinary 44%, Exceptional 33%, Elite 17%, Unique 6%
  - Card average prices are computed per set + rarity + finish, excluding sealed products (booster/box/case/pack/etc).
  - Sealed prices are pulled from rows whose name contains "booster" + (pack|box|case).

Example:
  python sorcery_ev_calc.py 20260213_0132_sorcery_prices.csv
  python sorcery_ev_calc.py 20260213_0132_sorcery_prices.csv --prev 20260212_1400_sorcery_prices.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


SEALED_NAME_RE = re.compile(r"\b(?:booster|box|case|pack|display|starter|precon|deck)\b", re.I)


def norm_finish(x: object) -> str:
    s = str(x).strip().lower()
    return "foil" if "foil" in s else "standard"


def norm_rarity(x: object) -> str:
    s = str(x).strip().lower()
    if s in ("nan", "none", ""):
        return ""
    m = {
        "ordinary": "Ordinary",
        "exceptional": "Exceptional",
        "elite": "Elite",
        "unique": "Unique",
        "promo": "Promo",
    }
    return m.get(s, s.title())


def sealed_kind(name: object) -> Optional[str]:
    n = str(name).lower()
    if "booster" not in n:
        return None
    # prioritize explicit container word
    if "case" in n:
        return "case"
    if "box" in n:
        return "box"
    if "pack" in n:
        return "pack"
    return None


@dataclass
class Odds:
    packs_per_box: int = 36
    ordinary_per_pack: int = 11
    exceptional_per_pack: int = 3
    unique_rate: float = 1 / 5
    foil_rate: float = 1 / 4
    foil_dist: Dict[str, float] = None

    def __post_init__(self):
        if self.foil_dist is None:
            self.foil_dist = {
                "Ordinary": 0.44,
                "Exceptional": 0.33,
                "Elite": 0.17,
                "Unique": 0.06,
            }


def compute_ev_table(df: pd.DataFrame, odds: Odds, require_box_price: bool = True) -> pd.DataFrame:
    # Required columns check (soft)
    for col in ("name", "price", "expansion", "finish", "rarity"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["expansion"] = df["expansion"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["finish_n"] = df["finish"].map(norm_finish)
    df["rarity_n"] = df["rarity"].map(norm_rarity)
    df["sealed_kind"] = df["name"].map(sealed_kind)

    # Sealed prices per set (take max in case multiple retailers/listings exist)
    sealed = (
        df[df["sealed_kind"].notna() & (df["finish_n"] == "standard") & df["price"].notna()]
        .groupby(["expansion", "sealed_kind"])["price"]
        .max()
        .unstack()
    )

    # Card prices (exclude sealed-ish products)
    cards = df[~df["name"].str.contains(SEALED_NAME_RE, regex=True) & df["price"].notna()]

    avg = (
        cards[cards["rarity_n"].isin(["Ordinary", "Exceptional", "Elite", "Unique"])]
        .groupby(["expansion", "rarity_n", "finish_n"])["price"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    def get_avg(exp: str, rarity: str, finish: str) -> float:
        try:
            v = avg.loc[(exp, rarity), finish]
            return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    rows = []
    for exp in sorted(df["expansion"].dropna().unique()):
        # require at least some card data
        o = get_avg(exp, "Ordinary", "standard")
        e = get_avg(exp, "Exceptional", "standard")
        el = get_avg(exp, "Elite", "standard")
        u = get_avg(exp, "Unique", "standard")
        if all(pd.isna(x) for x in (o, e, el, u)):
            continue

        rare_ev = odds.unique_rate * (0 if pd.isna(u) else u) + (1 - odds.unique_rate) * (0 if pd.isna(el) else el)
        nonfoil_ev = (
            odds.ordinary_per_pack * (0 if pd.isna(o) else o)
            + odds.exceptional_per_pack * (0 if pd.isna(e) else e)
            + rare_ev
        )

        foil_ev_single = 0.0
        for r, p in odds.foil_dist.items():
            fv = get_avg(exp, r, "foil")
            foil_ev_single += p * (0 if pd.isna(fv) else fv)
        foil_ev_pack = odds.foil_rate * foil_ev_single

        pack_ev = nonfoil_ev + foil_ev_pack
        box_ev = pack_ev * odds.packs_per_box

        pack_price = sealed.loc[exp, "pack"] if (exp in sealed.index and "pack" in sealed.columns) else np.nan
        box_price = sealed.loc[exp, "box"] if (exp in sealed.index and "box" in sealed.columns) else np.nan
        case_price = sealed.loc[exp, "case"] if (exp in sealed.index and "case" in sealed.columns) else np.nan

        rows.append(
            {
                "Set": exp,
                "Pack EV ($)": pack_ev,
                "Box EV ($)": box_ev,
                "Pack Price ($)": pack_price,
                "Box Price ($)": box_price,
                "Case Price ($)": case_price,
                "EV - Box Price ($)": (box_ev - box_price) if pd.notna(box_price) else np.nan,
                "Box EV / Box Price": (box_ev / box_price) if (pd.notna(box_price) and box_price != 0) else np.nan,
                "Assumption: packs/box": odds.packs_per_box,
                "Assumption: unique_rate": odds.unique_rate,
                "Assumption: foil_rate": odds.foil_rate,
            }
        )

    out = pd.DataFrame(rows)

    if require_box_price:
        out = out[pd.notna(out["Box Price ($)"])]

    out = out.sort_values("Box EV ($)", ascending=False)

    return out


# ── Biggest changes since last run ────────────────────────────────────


def find_previous_csv(current_csv: str) -> Optional[str]:
    """Auto-detect the most recent sorcery_prices CSV before the current one."""
    directory = os.path.dirname(os.path.abspath(current_csv)) or "."
    pattern = os.path.join(directory, "*_sorcery_prices.csv")
    candidates = sorted(glob.glob(pattern))
    current_abs = os.path.abspath(current_csv)
    # Filter out the current file and pick the most recent remaining one
    candidates = [c for c in candidates if os.path.abspath(c) != current_abs]
    return candidates[-1] if candidates else None


def compute_price_changes(current_df: pd.DataFrame, prev_df: pd.DataFrame,
                          top_n: int = 20) -> pd.DataFrame:
    """Compare two price CSVs and return the biggest movers.

    Returns a DataFrame with columns:
      name, expansion, finish, prev_price, curr_price, change ($), change (%), art_link
    sorted by absolute dollar change descending.
    """
    for df in (current_df, prev_df):
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Build a merge key: (name, expansion, finish)
    curr = current_df[current_df["price"].notna()].copy()
    prev = prev_df[prev_df["price"].notna()].copy()

    merged = curr.merge(
        prev[["name", "expansion", "finish", "price"]],
        on=["name", "expansion", "finish"],
        how="inner",
        suffixes=("_curr", "_prev"),
    )

    merged["change ($)"] = merged["price_curr"] - merged["price_prev"]
    merged["change (%)"] = merged.apply(
        lambda r: (r["change ($)"] / r["price_prev"] * 100) if r["price_prev"] != 0 else np.nan,
        axis=1,
    )
    merged["abs_change"] = merged["change ($)"].abs()

    # Sort by absolute dollar change, take top N
    merged = merged.sort_values("abs_change", ascending=False).head(top_n)

    out_cols = ["name", "expansion", "finish"]
    if "rarity" in merged.columns:
        out_cols.append("rarity")
    out_cols += ["price_prev", "price_curr", "change ($)", "change (%)"]
    if "art_link" in merged.columns:
        out_cols.append("art_link")

    result = merged[out_cols].copy()
    result = result.rename(columns={"price_prev": "Prev Price ($)", "price_curr": "Curr Price ($)"})
    return result


def print_changes_report(changes: pd.DataFrame) -> None:
    """Pretty-print the biggest price changes."""
    if changes.empty:
        print("\nNo price changes detected (or no overlapping cards found).")
        return

    print("\n" + "=" * 70)
    print("  BIGGEST PRICE CHANGES SINCE LAST RUN")
    print("=" * 70)

    gainers = changes[changes["change ($)"] > 0]
    losers = changes[changes["change ($)"] < 0]

    if not gainers.empty:
        print("\n  TOP GAINERS:")
        print("  " + "-" * 66)
        for _, row in gainers.iterrows():
            pct = f"{row['change (%)']:+.1f}%" if pd.notna(row["change (%)"]) else "N/A"
            print(f"  {row['name']:<35} {row['expansion']:<18} "
                  f"${row['Prev Price ($)']:>8.2f} -> ${row['Curr Price ($)']:>8.2f}  "
                  f"({pct})")

    if not losers.empty:
        print("\n  TOP LOSERS:")
        print("  " + "-" * 66)
        for _, row in losers.iterrows():
            pct = f"{row['change (%)']:+.1f}%" if pd.notna(row["change (%)"]) else "N/A"
            print(f"  {row['name']:<35} {row['expansion']:<18} "
                  f"${row['Prev Price ($)']:>8.2f} -> ${row['Curr Price ($)']:>8.2f}  "
                  f"({pct})")

    print()

    # Summary stats
    all_changes = changes["change ($)"]
    print(f"  Cards compared: {len(changes)}")
    print(f"  Avg absolute change: ${all_changes.abs().mean():.2f}")
    print(f"  Max gain:  ${all_changes.max():.2f}")
    print(f"  Max loss:  ${all_changes.min():.2f}")
    print("=" * 70)


# ── Interesting calculations ──────────────────────────────────────────


def print_interesting_stats(df: pd.DataFrame) -> None:
    """Print additional interesting calculations from the price data."""
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Exclude sealed products
    cards = df[~df["name"].str.contains(SEALED_NAME_RE, regex=True) & df["price"].notna()].copy()
    cards["finish_n"] = cards["finish"].map(norm_finish)
    cards["rarity_n"] = cards["rarity"].map(norm_rarity)

    print("\n" + "=" * 70)
    print("  INTERESTING STATS")
    print("=" * 70)

    # Foil multiplier by rarity
    print("\n  FOIL MULTIPLIER BY RARITY (avg foil price / avg standard price):")
    print("  " + "-" * 66)
    for rarity in ["Ordinary", "Exceptional", "Elite", "Unique"]:
        std = cards[(cards["rarity_n"] == rarity) & (cards["finish_n"] == "standard")]["price"]
        foil = cards[(cards["rarity_n"] == rarity) & (cards["finish_n"] == "foil")]["price"]
        if len(std) > 0 and len(foil) > 0 and std.mean() > 0:
            mult = foil.mean() / std.mean()
            print(f"    {rarity:<14} {mult:>6.1f}x  "
                  f"(std avg ${std.mean():.2f}, foil avg ${foil.mean():.2f})")

    # Most expensive cards per set
    print("\n  TOP 5 MOST EXPENSIVE CARDS PER SET:")
    print("  " + "-" * 66)
    for exp in sorted(cards["expansion"].unique()):
        set_cards = cards[cards["expansion"] == exp].nlargest(5, "price")
        print(f"\n    {exp}:")
        for _, row in set_cards.iterrows():
            finish_tag = " [Foil]" if row["finish_n"] == "foil" else ""
            print(f"      ${row['price']:>8.2f}  {row['name']}{finish_tag}")

    # Price distribution by rarity
    print("\n  PRICE DISTRIBUTION BY RARITY (standard finish):")
    print("  " + "-" * 66)
    std_cards = cards[cards["finish_n"] == "standard"]
    for rarity in ["Ordinary", "Exceptional", "Elite", "Unique"]:
        r_cards = std_cards[std_cards["rarity_n"] == rarity]["price"]
        if len(r_cards) > 0:
            print(f"    {rarity:<14} count={len(r_cards):<4}  "
                  f"min=${r_cards.min():.2f}  median=${r_cards.median():.2f}  "
                  f"mean=${r_cards.mean():.2f}  max=${r_cards.max():.2f}")

    # Total portfolio value by set
    print("\n  TOTAL CARD VALUE BY SET (1x each card, standard only):")
    print("  " + "-" * 66)
    std_totals = (
        std_cards.groupby("expansion")["price"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
    )
    for exp, row in std_totals.iterrows():
        print(f"    {exp:<25} ${row['sum']:>10.2f}  ({int(row['count'])} cards)")

    # Unique-to-Elite price ratio by set
    print("\n  UNIQUE / ELITE AVERAGE PRICE RATIO BY SET (standard):")
    print("  " + "-" * 66)
    for exp in sorted(cards["expansion"].unique()):
        elite = std_cards[(std_cards["expansion"] == exp) & (std_cards["rarity_n"] == "Elite")]["price"]
        unique = std_cards[(std_cards["expansion"] == exp) & (std_cards["rarity_n"] == "Unique")]["price"]
        if len(elite) > 0 and len(unique) > 0 and elite.mean() > 0:
            ratio = unique.mean() / elite.mean()
            print(f"    {exp:<25} {ratio:>6.1f}x  "
                  f"(elite avg ${elite.mean():.2f}, unique avg ${unique.mean():.2f})")

    print("\n" + "=" * 70)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Compute rough Sorcery EV by set using a community-odds model.")
    p.add_argument("csv", help="Path to sorcery_prices CSV")
    p.add_argument("--prev", help="Path to previous sorcery_prices CSV (auto-detected if omitted)")
    p.add_argument("--top-n", type=int, default=20, help="Number of biggest movers to show (default: 20)")
    p.add_argument("--include-sets-without-box-price", action="store_true", help="Include sets even if no box price found in CSV")
    p.add_argument("--packs-per-box", type=int, default=36)
    p.add_argument("--ordinary-per-pack", type=int, default=11)
    p.add_argument("--exceptional-per-pack", type=int, default=3)
    p.add_argument("--unique-rate", type=float, default=1/5, help="Probability that the rare slot is Unique (else Elite)")
    p.add_argument("--foil-rate", type=float, default=1/4, help="Expected foils per pack")
    p.add_argument("--foil-dist", type=str, default="", help='JSON dict like {"Ordinary":0.44,"Exceptional":0.33,"Elite":0.17,"Unique":0.06}')
    p.add_argument("--no-stats", action="store_true", help="Skip interesting stats output")
    args = p.parse_args(argv)

    foil_dist = None
    if args.foil_dist.strip():
        import json
        foil_dist = json.loads(args.foil_dist)

    odds = Odds(
        packs_per_box=args.packs_per_box,
        ordinary_per_pack=args.ordinary_per_pack,
        exceptional_per_pack=args.exceptional_per_pack,
        unique_rate=args.unique_rate,
        foil_rate=args.foil_rate,
        foil_dist=foil_dist,
    )

    df = pd.read_csv(args.csv)

    # Timestamped output filenames — never overwrite old data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.dirname(os.path.abspath(args.csv)) or "."
    ev_out = os.path.join(out_dir, f"{timestamp}_ev_table.csv")
    changes_out = os.path.join(out_dir, f"{timestamp}_changes.csv")

    # ── EV Table ──────────────────────────────────────────────────────
    out = compute_ev_table(df, odds, require_box_price=not args.include_sets_without_box_price)

    # Pretty print
    show = out.copy()
    for c in ["Pack EV ($)", "Box EV ($)", "Pack Price ($)", "Box Price ($)", "Case Price ($)", "EV - Box Price ($)", "Box EV / Box Price"]:
        if c in show.columns:
            show[c] = show[c].astype(float).round(3 if c == "Box EV / Box Price" else 2)
    cols = ["Set", "Pack EV ($)", "Box EV ($)", "Pack Price ($)", "Box Price ($)", "Case Price ($)", "EV - Box Price ($)", "Box EV / Box Price"]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].to_string(index=False))

    out.to_csv(ev_out, index=False)
    print(f"\nWrote EV table: {ev_out}")

    # ── Biggest changes since last run ────────────────────────────────
    prev_csv = args.prev
    if not prev_csv:
        prev_csv = find_previous_csv(args.csv)

    if prev_csv and os.path.exists(prev_csv):
        print(f"\nComparing against previous run: {os.path.basename(prev_csv)}")
        prev_df = pd.read_csv(prev_csv)
        changes = compute_price_changes(df, prev_df, top_n=args.top_n)
        print_changes_report(changes)

        changes.to_csv(changes_out, index=False)
        print(f"Wrote changes: {changes_out}")
    else:
        print("\nNo previous CSV found for comparison. Run the scraper again later to see price changes.")

    # ── Interesting stats ─────────────────────────────────────────────
    if not args.no_stats:
        print_interesting_stats(df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

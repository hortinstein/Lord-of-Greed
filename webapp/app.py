#!/usr/bin/env python3
"""Flask web app for browsing Sorcery TCG card prices."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

import pandas as pd
import requests as http_requests
from flask import Flask, jsonify, render_template, request, Response

app = Flask(__name__)

# ── Paths & constants ────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, os.pardir, "data")

TS_RE = re.compile(r"^(\d{8}_\d{4})_sorcery_prices\.csv$")
SEALED_NAME_RE = re.compile(
    r"\b(?:booster|box|case|pack|display|starter|precon|deck)\b", re.I
)
SEALED_CASE_RE = re.compile(r"\b(?:box\s*case|booster\s*case)\b|case\b", re.I)
SEALED_BOX_RE = re.compile(r"\b(?:booster\s*box)\b", re.I)
SEALED_PLEDGE_RE = re.compile(r"\bpledge\s*pack\b", re.I)
SEALED_PACK_RE = re.compile(r"\b(?:booster\s*pack|pack)\b", re.I)
SEALED_DECK_RE = re.compile(r"\b(?:precon|deck)\b", re.I)
PRODUCT_ID_RE = re.compile(r"/product/(\d+)")
TCGPLAYER_CDN = "https://tcgplayer-cdn.tcgplayer.com/product/{pid}_in_1000x1000.jpg"

# ── Data loading (runs once at startup) ──────────────────────────────

LABELS: list[str] = []
ALL_DF: pd.DataFrame = pd.DataFrame()
LATEST_DF: pd.DataFrame = pd.DataFrame()
FILTER_OPTIONS: dict = {}


def _is_sealed(name: str) -> bool:
    return bool(SEALED_NAME_RE.search(str(name)))


def _sealed_type(name: str) -> str:
    """Classify a sealed product into Case, Box, Pack, Pledge Pack, or Deck."""
    n = str(name)
    if SEALED_CASE_RE.search(n):
        return "Case"
    if SEALED_BOX_RE.search(n):
        return "Box"
    if SEALED_PLEDGE_RE.search(n):
        return "Pledge Pack"
    if SEALED_PACK_RE.search(n):
        return "Pack"
    if SEALED_DECK_RE.search(n):
        return "Deck"
    if "box topper" in n.lower():
        return "Box Topper"
    return "Other"


def _image_url(art_link: str) -> str | None:
    m = PRODUCT_ID_RE.search(str(art_link))
    if m:
        return TCGPLAYER_CDN.format(pid=m.group(1))
    return None


def _discover_snapshots() -> list[tuple[str, str]]:
    pairs = []
    for fname in os.listdir(DATA_DIR):
        m = TS_RE.match(fname)
        if m:
            ts_raw = m.group(1)
            label = datetime.strptime(ts_raw, "%Y%m%d_%H%M").strftime("%m/%d %H:%M")
            pairs.append((label, os.path.join(DATA_DIR, fname)))
    pairs.sort()
    return pairs


def _load_data() -> None:
    global LABELS, ALL_DF, LATEST_DF, FILTER_OPTIONS

    snapshots = _discover_snapshots()
    if not snapshots:
        print("WARNING: No CSV snapshots found in data/")
        return

    frames = []
    labels = []
    for label, path in snapshots:
        df = pd.read_csv(path)
        df["snapshot"] = label
        frames.append(df)
        labels.append(label)

    combined = pd.concat(frames, ignore_index=True)
    combined["price"] = pd.to_numeric(combined["price"], errors="coerce")

    # Tag sealed vs cards
    combined["category"] = combined["name"].apply(
        lambda n: "Sealed" if _is_sealed(n) else "Cards"
    )
    combined["sealed_type"] = combined["name"].apply(
        lambda n: _sealed_type(n) if _is_sealed(n) else ""
    )

    LABELS = labels
    ALL_DF = combined

    # Latest snapshot for the card grid
    latest_snap = labels[-1]
    latest = combined[combined["snapshot"] == latest_snap].copy()
    latest["image_url"] = latest["art_link"].apply(_image_url)
    latest["_name_norm"] = latest["name"].apply(_normalize_name)
    LATEST_DF = latest.reset_index(drop=True)

    # Filter options
    sealed_types = LATEST_DF.loc[
        LATEST_DF["sealed_type"] != "", "sealed_type"
    ].dropna().unique().tolist()
    FILTER_OPTIONS = {
        "categories": sorted(LATEST_DF["category"].dropna().unique().tolist()),
        "sealed_types": sorted(sealed_types),
        "rarities": sorted(LATEST_DF["rarity"].dropna().unique().tolist()),
        "expansions": sorted(LATEST_DF["expansion"].dropna().unique().tolist()),
        "finishes": sorted(LATEST_DF["finish"].dropna().unique().tolist()),
    }


# ── Routes ───────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/filters")
def api_filters():
    return jsonify(FILTER_OPTIONS)


@app.route("/api/cards")
def api_cards():
    df = LATEST_DF.copy()

    # Search
    search = request.args.get("search", "").strip()
    if search:
        df = df[df["name"].str.contains(search, case=False, na=False)]

    # Filters
    category = request.args.get("category", "").strip()
    if category:
        df = df[df["category"] == category]

    rarity = request.args.get("rarity", "").strip()
    if rarity:
        df = df[df["rarity"] == rarity]

    expansion = request.args.get("expansion", "").strip()
    if expansion:
        df = df[df["expansion"] == expansion]

    finish = request.args.get("finish", "").strip()
    if finish:
        df = df[df["finish"] == finish]

    sealed_type = request.args.get("sealed_type", "").strip()
    if sealed_type:
        df = df[df["sealed_type"] == sealed_type]

    # Sort
    sort_by = request.args.get("sort", "name").strip()
    order = request.args.get("order", "asc").strip()
    ascending = order != "desc"

    RARITY_ORDER = {"Ordinary": 0, "Exceptional": 1, "Elite": 2, "Unique": 3, "Promo": 4}

    df = df.copy()
    if sort_by == "rarity":
        df["_sort_key"] = df["rarity"].map(RARITY_ORDER).fillna(99)
        df = df.sort_values(["_sort_key", "name"], ascending=[ascending, True],
                            na_position="last")
        df = df.drop(columns=["_sort_key"])
    elif sort_by == "price":
        df = df.sort_values(["price", "name"], ascending=[ascending, True],
                            na_position="last")
    elif sort_by == "expansion":
        df = df.sort_values(["expansion", "name"], ascending=[ascending, True],
                            na_position="last")
    else:
        df = df.sort_values("name", ascending=ascending, na_position="last")

    # Pagination
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(1, int(request.args.get("per_page", 50))))
    total = len(df)
    start = (page - 1) * per_page
    page_df = df.iloc[start : start + per_page]

    cards = []
    for _, row in page_df.iterrows():
        cards.append(
            {
                "name": row["name"],
                "price": _safe(row["price"]),
                "expansion": _safe(row["expansion"]) or "",
                "finish": _safe(row["finish"]) or "",
                "rarity": _safe(row.get("rarity", "")) or "",
                "category": _safe(row.get("category", "Cards")) or "Cards",
                "sealed_type": _safe(row.get("sealed_type", "")) or "",
                "art_link": _safe(row.get("art_link", "")) or "",
                "image_url": _safe(row.get("image_url", "")) or "",
            }
        )

    return jsonify(
        {
            "cards": cards,
            "page": page,
            "per_page": per_page,
            "total": total,
            "has_more": start + per_page < total,
        }
    )


@app.route("/api/card/history")
def api_card_history():
    name = request.args.get("name", "").strip()
    expansion = request.args.get("expansion", "").strip()
    finish = request.args.get("finish", "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    mask = ALL_DF["name"] == name
    category = request.args.get("category", "").strip()
    if category:
        mask &= ALL_DF["category"] == category
    if expansion:
        mask &= ALL_DF["expansion"] == expansion
    if finish:
        mask &= ALL_DF["finish"] == finish

    subset = ALL_DF[mask]
    snap_prices = dict(zip(subset["snapshot"], subset["price"]))
    prices = [
        snap_prices.get(lbl) if pd.notna(snap_prices.get(lbl)) else None
        for lbl in LABELS
    ]

    return jsonify({"labels": LABELS, "prices": prices})


@app.route("/api/sealed/stats")
def api_sealed_stats():
    """Return aggregate stats for sealed products."""
    sealed = LATEST_DF[LATEST_DF["category"] == "Sealed"].copy()
    if sealed.empty:
        return jsonify({"count": 0, "products": []})

    sealed_sorted = sealed.sort_values("price", ascending=False, na_position="last")

    products = []
    for _, row in sealed_sorted.iterrows():
        products.append({
            "name": row["name"],
            "price": _safe(row["price"]),
            "expansion": _safe(row["expansion"]) or "",
            "finish": _safe(row["finish"]) or "",
            "rarity": _safe(row.get("rarity", "")) or "",
            "sealed_type": _safe(row.get("sealed_type", "")) or "",
            "image_url": _safe(row.get("image_url", "")) or "",
            "art_link": _safe(row.get("art_link", "")) or "",
        })

    prices = sealed["price"].dropna()
    by_expansion = (
        sealed.groupby("expansion")["price"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    expansion_stats = []
    for _, r in by_expansion.iterrows():
        expansion_stats.append({
            "expansion": r["expansion"],
            "avg_price": round(r["mean"], 2),
            "min_price": round(r["min"], 2),
            "max_price": round(r["max"], 2),
            "count": int(r["count"]),
        })

    # Price history for all sealed (average), plus per sealed_type
    sealed_all = ALL_DF[ALL_DF["category"] == "Sealed"].copy()
    sealed_all["sealed_type"] = sealed_all["name"].apply(
        lambda n: _sealed_type(n) if _is_sealed(n) else ""
    )
    avg_by_snap = sealed_all.groupby("snapshot")["price"].mean()
    history_prices = [
        round(avg_by_snap.get(lbl), 2) if lbl in avg_by_snap and pd.notna(avg_by_snap.get(lbl)) else None
        for lbl in LABELS
    ]

    # Per-type history for filtered chart
    history_by_type = {}
    for stype in sealed_all["sealed_type"].unique():
        if not stype:
            continue
        type_df = sealed_all[sealed_all["sealed_type"] == stype]
        type_avg = type_df.groupby("snapshot")["price"].mean()
        history_by_type[stype] = [
            round(type_avg.get(lbl), 2) if lbl in type_avg and pd.notna(type_avg.get(lbl)) else None
            for lbl in LABELS
        ]

    return jsonify({
        "count": len(sealed),
        "total_value": round(prices.sum(), 2) if not prices.empty else 0,
        "avg_price": round(prices.mean(), 2) if not prices.empty else 0,
        "min_price": round(prices.min(), 2) if not prices.empty else 0,
        "max_price": round(prices.max(), 2) if not prices.empty else 0,
        "by_expansion": expansion_stats,
        "products": products,
        "history_labels": LABELS,
        "history_avg_prices": history_prices,
        "history_by_type": history_by_type,
    })


CURIOSA_DECK_RE = re.compile(r"curiosa\.io/decks/([a-zA-Z0-9]+)")
CURIOSA_TRPC = "https://curiosa.io/api/trpc"
CURIOSA_HEADERS = {
    "Origin": "https://curiosa.io",
    "Referer": "https://curiosa.io/",
    "User-Agent": "Mozilla/5.0",
}


def _curiosa_fetch(procedure: str, deck_id: str) -> list | dict:
    """Call a Curiosa tRPC endpoint and return the JSON result."""
    import urllib.parse

    inp = json.dumps({"json": {"id": deck_id}})
    url = f"{CURIOSA_TRPC}/{procedure}?input={urllib.parse.quote(inp)}"
    resp = http_requests.get(url, headers=CURIOSA_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()["result"]["data"]["json"]


def _safe(val):
    """Convert pandas NaN/NaT to None, leave everything else."""
    if pd.isna(val):
        return None
    return val


def _normalize_name(name: str) -> str:
    """Strip curly quotes, hyphens, and other punctuation for fuzzy matching."""
    n = name.lower()
    # Curly apostrophes / quotes -> nothing (CSV omits them entirely)
    n = n.replace("\u2019", "").replace("\u2018", "")
    n = n.replace("\u201c", "").replace("\u201d", "")
    n = n.replace("'", "").replace("'", "")
    # Hyphens -> space (CSV uses spaces)
    n = n.replace("-", " ")
    # Collapse multiple spaces
    n = " ".join(n.split())
    return n


def _match_card_price(name: str, finish_pref: str) -> dict:
    """Look up a card name in our price data."""
    norm = _normalize_name(name)
    matches = LATEST_DF[LATEST_DF["_name_norm"] == norm]
    if finish_pref and not matches.empty:
        pref = matches[matches["finish"] == finish_pref]
        if not pref.empty:
            matches = pref
    if matches.empty:
        return {"found": False, "price": None, "expansion": "",
                "finish": "", "rarity": "", "image_url": "", "art_link": ""}
    row = matches.iloc[0]
    return {
        "found": True,
        "price": _safe(row["price"]),
        "expansion": _safe(row["expansion"]) or "",
        "finish": _safe(row["finish"]) or "",
        "rarity": _safe(row.get("rarity", "")) or "",
        "image_url": _safe(row.get("image_url", "")) or "",
        "art_link": _safe(row.get("art_link", "")) or "",
    }


@app.route("/api/deck", methods=["POST"])
def api_deck():
    """Fetch a Curiosa deck and return cards with local price data."""
    body = request.get_json(silent=True) or {}
    url = body.get("url", "").strip()
    finish_pref = body.get("finish", "Standard")

    m = CURIOSA_DECK_RE.search(url)
    if not m:
        return jsonify({"error": "Invalid Curiosa deck URL"}), 400

    deck_id = m.group(1)

    try:
        meta = _curiosa_fetch("deck.getById", deck_id)
        avatar_data = _curiosa_fetch("deck.getAvatarById", deck_id)
        decklist = _curiosa_fetch("deck.getDecklistById", deck_id)
        sideboard = _curiosa_fetch("deck.getSideboardById", deck_id)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch deck: {e}"}), 502

    def _build_entries(items: list) -> list[dict]:
        entries = []
        for item in items:
            card = item.get("card", {})
            name = card.get("name", "")
            qty = item.get("quantity", 1)
            price_info = _match_card_price(name, finish_pref)
            entries.append({
                "qty": qty,
                "name": name,
                "type": card.get("type", ""),
                "category": card.get("category", ""),
                **price_info,
            })
        return entries

    # Build avatar entry
    avatar_entry = None
    if isinstance(avatar_data, dict) and avatar_data.get("card"):
        ac = avatar_data["card"]
        ap = _match_card_price(ac.get("name", ""), finish_pref)
        avatar_entry = {"qty": 1, "name": ac.get("name", ""),
                        "type": "Avatar", "category": "Avatar", **ap}

    deck_entries = _build_entries(decklist)
    side_entries = _build_entries(sideboard)
    all_priced = ([avatar_entry] if avatar_entry else []) + deck_entries + side_entries

    total_price = sum(
        (e["price"] or 0) * e["qty"] for e in all_priced if e["found"]
    )

    return jsonify({
        "deck_name": meta.get("name", ""),
        "format": meta.get("format", ""),
        "avatar": avatar_entry,
        "decklist": deck_entries,
        "sideboard": side_entries,
        "total_price": total_price,
    })


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_data()
    print(f"Loaded {len(LABELS)} snapshots, {len(LATEST_DF)} cards in latest")
    app.run(debug=True, host="0.0.0.0", port=5000)

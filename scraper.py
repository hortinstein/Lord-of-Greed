#!/usr/bin/env python3
"""Sorcery TCG Price Scraper.

Combines card data from the Sorcery TCG API with market prices from
TCGPlayer (via TCGCSV proxy) and outputs a CSV with columns:
name, price, expansion, finish, rarity, art_link
"""

import csv
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

SORCERY_API_URL = "https://api.sorcerytcg.com/api/cards"
TCGCSV_BASE = "https://tcgcsv.com/tcgplayer/77"
OUTPUT_FILE = f"{datetime.now().strftime('%Y%m%d_%H%M')}_sorcery_prices.csv"

# TCGCSV group names that don't exist in the Sorcery API.
# Map them to the Sorcery API set name for variant lookup.
TCGCSV_TO_SORCERY_SET = {
    "Dust Reward Promos": "Promotional",
    "Arthurian Legends Promo": "Promotional",
}


def normalize(name: str) -> str:
    """Normalize a name for fuzzy matching: strip diacritics, punctuation, lowercase."""
    # Decompose unicode and strip combining marks (diacritics)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase, strip outer whitespace
    ascii_name = ascii_name.strip().lower()
    # Replace hyphens with spaces (TCGPlayer does this)
    ascii_name = ascii_name.replace("-", " ")
    # Remove remaining punctuation (apostrophes, etc.) but keep spaces
    ascii_name = re.sub(r"[^\w\s]", "", ascii_name)
    # Collapse multiple spaces
    ascii_name = re.sub(r"\s+", " ", ascii_name)
    return ascii_name


def strip_foil_suffix(name: str) -> str:
    """TCGPlayer appends ' Foil' to foil product names — strip it."""
    if name.endswith(" Foil"):
        return name[:-5]
    return name


# ── Step 1: Sorcery API ─────────────────────────────────────────────

def fetch_sorcery_lookup() -> dict:
    """Build lookup: (norm_name, norm_set, finish_lower) → {slug, rarity, artist}."""
    print("Fetching Sorcery API card data…")
    resp = requests.get(SORCERY_API_URL, timeout=60)
    resp.raise_for_status()
    cards = resp.json()
    print(f"  {len(cards)} cards returned")

    lookup: dict[tuple, dict] = {}
    for card in cards:
        name = card.get("name", "")
        rarity = card.get("guardian", {}).get("rarity", "")
        for card_set in card.get("sets", []):
            set_name = card_set.get("name", "")
            for variant in card_set.get("variants", []):
                slug = variant.get("slug", "")
                finish = variant.get("finish", "")  # "Standard" / "Foil"
                key = (normalize(name), normalize(set_name), finish.lower())
                lookup[key] = {
                    "slug": slug,
                    "rarity": rarity,
                    "artist": variant.get("artist", ""),
                }
    print(f"  {len(lookup)} variant entries indexed")
    return lookup


# ── Step 2: TCGCSV (TCGPlayer proxy) ────────────────────────────────

def fetch_groups() -> list[dict]:
    """Fetch all Sorcery TCG groups (expansions) from TCGCSV."""
    print("Fetching TCGCSV groups…")
    resp = requests.get(f"{TCGCSV_BASE}/groups", timeout=30)
    resp.raise_for_status()
    groups = resp.json().get("results", [])
    for g in groups:
        print(f"  {g['groupId']}: {g['name']}")
    return groups


def fetch_group_products_and_prices(group_id: int, group_name: str):
    """Fetch products + prices for one group. Returns (name, products, prices)."""
    prod_resp = requests.get(f"{TCGCSV_BASE}/{group_id}/products", timeout=60)
    prod_resp.raise_for_status()
    products = prod_resp.json().get("results", [])

    price_resp = requests.get(f"{TCGCSV_BASE}/{group_id}/prices", timeout=60)
    price_resp.raise_for_status()
    prices = price_resp.json().get("results", [])

    return group_name, products, prices


def rarity_from_extended(extended_data: list) -> str:
    for item in extended_data:
        if item.get("name") == "Rarity":
            return item.get("value", "")
    return ""


# ── Step 3: Merge ────────────────────────────────────────────────────

def build_rows(groups: list[dict], sorcery_lookup: dict) -> list[dict]:
    """Fetch TCGCSV data for every group, merge with Sorcery lookup."""
    rows: list[dict] = []

    print("Fetching products & prices for all groups…")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_group_products_and_prices, g["groupId"], g["name"]): g
            for g in groups
        }
        results = []
        for future in as_completed(futures):
            gname, products, prices = future.result()
            results.append((gname, products, prices))
            print(f"  {gname}: {len(products)} products, {len(prices)} price entries")

    for group_name, products, prices in results:
        product_by_id = {p["productId"]: p for p in products}

        # Group prices by productId (one product can have Normal + Foil)
        prices_by_id: dict[int, list] = {}
        for pr in prices:
            prices_by_id.setdefault(pr["productId"], []).append(pr)

        # Determine which Sorcery API set name to use for lookup
        sorcery_set = TCGCSV_TO_SORCERY_SET.get(group_name, group_name)

        for pid, price_entries in prices_by_id.items():
            product = product_by_id.get(pid)
            if not product:
                continue

            card_name = product.get("cleanName") or product.get("name", "")
            tcg_rarity = rarity_from_extended(product.get("extendedData", []))

            for pe in price_entries:
                market_price = pe.get("marketPrice")
                if market_price is None:
                    continue

                sub_type = pe.get("subTypeName", "Normal")
                finish = "Foil" if sub_type == "Foil" else "Standard"

                # Strip " Foil" suffix that TCGPlayer appends to foil product names
                match_name = strip_foil_suffix(card_name) if finish == "Foil" else card_name

                # Try Sorcery API lookup
                key = (normalize(match_name), normalize(sorcery_set), finish.lower())
                variant = sorcery_lookup.get(key, {})

                rarity = variant.get("rarity") or tcg_rarity
                slug = variant.get("slug", "")
                art_link = slug  # slug doubles as the art identifier

                rows.append({
                    "name": card_name,
                    "price": market_price,
                    "expansion": group_name,
                    "finish": finish,
                    "rarity": rarity,
                    "art_link": art_link,
                })

    return rows


# ── Step 4: Write CSV ────────────────────────────────────────────────

def write_csv(rows: list[dict]) -> None:
    fieldnames = ["name", "price", "expansion", "finish", "rarity", "art_link"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_FILE}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    try:
        sorcery_lookup = fetch_sorcery_lookup()
        groups = fetch_groups()
        rows = build_rows(groups, sorcery_lookup)
        rows.sort(key=lambda r: (r["expansion"], r["name"], r["finish"]))
        write_csv(rows)

        # Quick stats
        matched = sum(1 for r in rows if r["art_link"])
        print(f"  {matched}/{len(rows)} rows matched to Sorcery API variants")
    except requests.RequestException as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe FEWS NET Data Warehouse for IPC coverage pre-2015.

Pulls the *full* historical IPC series for KE/ET/SO from
``fdw.fews.net/api/ipcphase/`` (no date filter in the URL) and reports
per-year / per-country row counts, Phase distribution (highlighting
Phase 5), and admin-1 coverage percentage. Informs the decision to
extend the panel backward from 2015 to 2009 (or 2012, or abandon).

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    venv/bin/python scripts/probe_ipc_history.py
"""

from __future__ import annotations

import io
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import requests

from src.config import ALL_REGIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("probe_ipc")

TIMEOUT = 180
PROBE_START = "2009-01-01"
PROBE_END = "2014-12-31"


def pull_country_full_history(country_code: str) -> pd.DataFrame:
    """Pull every IPC row FEWS NET has for a country, all years."""
    url = (
        f"https://fdw.fews.net/api/ipcphase/"
        f"?country={country_code}&format=csv&fields=simple"
    )
    logger.info("GET %s", url)
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df["country_code_2"] = country_code
    logger.info("  %s: %d raw rows, cols=%s", country_code, len(df), list(df.columns)[:8])
    return df


def vectorized_admin1_match(
    full_names: pd.Series, countries: pd.Series
) -> pd.Series:
    """Vectorized fuzzy match of geographic_unit_full_name to region codes.

    Uses str.extract on a single alternation regex built from ALL_REGIONS.
    Returns region_code or NaN.
    """
    prefix_map = {"KE": "KE", "ET": "ET", "SO": "SO"}

    # Build country -> [(region_code, name_lower)] map
    regions_by_country: dict[str, list[tuple[str, str]]] = {"KE": [], "ET": [], "SO": []}
    for code, name in ALL_REGIONS.items():
        prefix = code[:2]
        regions_by_country.setdefault(prefix, []).append((code, name.lower()))

    # Sort longer names first so "North Wollo" wins over "Wollo"
    for k in regions_by_country:
        regions_by_country[k].sort(key=lambda t: -len(t[1]))

    # Pre-lowercase the full names
    fn_lower = full_names.fillna("").astype(str).str.lower()

    # Build a per-country match vector
    out = pd.Series([None] * len(full_names), index=full_names.index, dtype=object)
    for country, regions in regions_by_country.items():
        mask = countries == country
        if not mask.any():
            continue
        subset = fn_lower[mask]
        # Try each region name (longest first) as a substring
        matched = pd.Series([None] * len(subset), index=subset.index, dtype=object)
        remaining = subset.copy()
        for code, name_lower in regions:
            hit = remaining.str.contains(name_lower, regex=False, na=False)
            if hit.any():
                matched.loc[hit[hit].index] = code
                remaining = remaining[~hit]
                if remaining.empty:
                    break
        out.loc[mask] = matched
    return out


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["reporting_date"] = pd.to_datetime(raw["reporting_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    cs = raw[raw["scenario"] == "CS"].copy()
    cs = cs.dropna(subset=["value", "reporting_date"])
    cs["ipc_phase"] = cs["value"].astype(int).clip(1, 5)
    cs["year"] = cs["reporting_date"].dt.year
    cs["month"] = cs["reporting_date"].dt.month

    cs["region_code"] = vectorized_admin1_match(
        cs["geographic_unit_full_name"], cs["country_code_2"]
    )
    logger.info(
        "Normalized: %d CS rows, %d admin-1 matched",
        len(cs),
        cs["region_code"].notna().sum(),
    )
    return cs


def report(cs: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("FEWS NET IPC — FULL HISTORICAL COVERAGE (Horn of Africa)")
    print("=" * 78)

    earliest = cs["reporting_date"].min()
    latest = cs["reporting_date"].max()
    total = len(cs)
    matched = cs["region_code"].notna().sum()
    print(f"Total CS rows: {total:>7,}  (admin-1 matched: {matched:>7,})")
    print(f"Earliest date: {earliest}")
    print(f"Latest date:   {latest}")

    print("\n-- Rows per year x country (admin-1 matched only) --")
    by_year_country = (
        cs.dropna(subset=["region_code"])
        .groupby(["year", "country_code_2"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(by_year_country.to_string())

    print("\n-- Phase distribution per year --")
    phase_by_year = (
        cs.dropna(subset=["region_code"])
        .groupby(["year", "ipc_phase"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(phase_by_year.to_string())

    p5 = cs[(cs["ipc_phase"] == 5) & cs["region_code"].notna()].copy()
    print(f"\n-- Phase 5 events (admin-1 matched): {len(p5)} rows --")
    if not p5.empty:
        p5_detail = (
            p5.groupby(["year", "country_code_2", "region_code"])
            .size()
            .reset_index(name="n")
            .sort_values(["year", "country_code_2"])
        )
        print(p5_detail.to_string(index=False))
    # Also check raw (unmatched) Phase-5
    p5_raw = cs[cs["ipc_phase"] == 5]
    print(f"\n-- Phase 5 (all CS rows incl. unmatched): {len(p5_raw)} rows --")
    if not p5_raw.empty:
        print(p5_raw.groupby(["year", "country_code_2"]).size().to_string())

    probe_start = pd.Timestamp(PROBE_START)
    probe_end = pd.Timestamp(PROBE_END)
    probe = cs[
        (cs["reporting_date"] >= probe_start)
        & (cs["reporting_date"] <= probe_end)
        & cs["region_code"].notna()
    ].copy()
    probe["ym"] = probe["reporting_date"].dt.to_period("M").dt.to_timestamp()

    n_regions = len(ALL_REGIONS)  # 37
    months = pd.date_range(probe_start, probe_end, freq="MS")
    cells_total = n_regions * len(months)

    observed = probe.drop_duplicates(subset=["region_code", "ym"])
    pct_observed = 100.0 * len(observed) / cells_total if cells_total else 0.0

    # After forward-fill (between analyses) — upper-bound coverage
    ff_cells = 0
    for rc in probe["region_code"].unique():
        rc_df = probe[probe["region_code"] == rc].sort_values("ym")
        if rc_df.empty:
            continue
        first_obs = rc_df["ym"].min()
        covered_months = months[(months >= first_obs) & (months <= probe_end)]
        ff_cells += len(covered_months)
    pct_ff = 100.0 * ff_cells / cells_total if cells_total else 0.0

    print(f"\n-- Admin-1 coverage for probe window {PROBE_START}..{PROBE_END} --")
    print(f"Raw observations:        {len(observed):>6,} / {cells_total:,} cells ({pct_observed:.1f}%)")
    print(f"After forward-fill:      {ff_cells:>6,} / {cells_total:,} cells ({pct_ff:.1f}%)")

    print("\n-- Per-country probe-window coverage --")
    for country in ["KE", "ET", "SO"]:
        c_regions = [code for code in ALL_REGIONS if code.startswith(country)]
        country_probe = probe[probe["country_code_2"] == country]
        n_country_regions = len(c_regions)
        country_cells = n_country_regions * len(months)
        country_observed = country_probe.drop_duplicates(subset=["region_code", "ym"])
        pct = (
            100.0 * len(country_observed) / country_cells
            if country_cells
            else 0.0
        )
        print(
            f"  {country}: {n_country_regions} regions × {len(months)} months "
            f"= {country_cells:,} cells; observed {len(country_observed):,} ({pct:.1f}%)"
        )

    print("\n-- Decision gate --")
    if pct_ff >= 80:
        rec = "PROCEED with 2009-01-01 start"
    elif pct_ff >= 60:
        rec = "FALL BACK to 2012-01-01 start"
    else:
        rec = "ABANDON extension; stay at 2015-01-01"
    print(f"  Forward-fill coverage {pct_ff:.1f}% -> {rec}")


def main() -> None:
    cache_path = "/tmp/probe_ipc_raw.parquet"
    if os.path.exists(cache_path):
        logger.info("Loading cached raw pulls from %s", cache_path)
        raw = pd.read_parquet(cache_path)
    else:
        frames = []
        for cc in ["KE", "ET", "SO"]:
            try:
                frames.append(pull_country_full_history(cc))
            except Exception as e:
                logger.error("Pull failed for %s: %s", cc, e)
        if not frames:
            logger.error("No data retrieved. Aborting.")
            sys.exit(1)
        raw = pd.concat(frames, ignore_index=True)
        raw.to_parquet(cache_path, index=False)
        logger.info("Cached raw to %s", cache_path)

    logger.info(
        "Fetched %d raw rows across %d countries.",
        len(raw),
        raw["country_code_2"].nunique(),
    )
    cs = normalize(raw)
    report(cs)


if __name__ == "__main__":
    main()

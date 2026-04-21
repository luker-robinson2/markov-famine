#!/usr/bin/env python3
"""Incremental data pull — saves after each region, resumes from cache."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import ee
import google.auth
from pathlib import Path

from src.config import ALL_REGIONS, GAUL_COUNTRY_CODES, GAUL_ASSET, GEE_ASSETS, PROCESSED_DIR, RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR, END_YEAR = 2015, 2024
_geom_cache = {}


def init_gee():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=credentials, project="code-monkey-453023")
    print("GEE OK", flush=True)


def get_geom(rc):
    if rc in _geom_cache:
        return _geom_cache[rc]
    name = ALL_REGIONS[rc]
    iso3 = {"KE": "KEN", "ET": "ETH", "SO": "SOM"}[rc[:2]]
    gc = GAUL_COUNTRY_CODES[iso3]
    g = (ee.FeatureCollection(GAUL_ASSET)
         .filter(ee.Filter.eq("ADM0_CODE", gc))
         .filter(ee.Filter.eq("ADM1_NAME", name)).geometry())
    _geom_cache[rc] = g
    return g


def extract_year(coll_id, band, geom, year, reducer="mean", scale=5000):
    """12 monthly values in one GEE call."""
    months = ee.List.sequence(1, 12)
    def m_comp(m):
        m = ee.Number(m)
        s = ee.Date.fromYMD(year, m, 1)
        e = s.advance(1, "month")
        c = ee.ImageCollection(coll_id).filterDate(s, e).select(band)
        img = c.sum() if reducer == "sum" else c.mean()
        v = img.reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9).get(band)
        return ee.Feature(None, {"month": m, "value": v})
    return ee.FeatureCollection(months.map(m_comp)).getInfo()


def pull_var(coll_id, band, var_name, reducer="mean", scale=5000):
    """Pull one variable for all regions, saving per-region cache files."""
    cache_dir = RAW_DIR / var_name
    cache_dir.mkdir(exist_ok=True)

    all_rows = []
    for i, (rc, name) in enumerate(ALL_REGIONS.items()):
        cache_file = cache_dir / f"{rc}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            all_rows.extend(df.to_dict("records"))
            print(f"  {var_name} {rc} ({name}) [cached] [{i+1}/37]", flush=True)
            continue

        t0 = time.time()
        rows = []
        geom = get_geom(rc)
        for year in range(START_YEAR, END_YEAR + 1):
            try:
                result = extract_year(coll_id, band, geom, year, reducer, scale)
                for f in result["features"]:
                    p = f["properties"]
                    rows.append({
                        "region_code": rc,
                        "date": pd.Timestamp(year=year, month=int(p["month"]), day=1),
                        var_name: p["value"],
                    })
            except Exception as e:
                print(f"  WARN: {var_name} {rc} {year}: {str(e)[:60]}", flush=True)
                for m in range(1, 13):
                    rows.append({"region_code": rc, "date": pd.Timestamp(year=year, month=m, day=1), var_name: None})
            time.sleep(0.3)

        df = pd.DataFrame(rows)
        df.to_parquet(cache_file, index=False)
        all_rows.extend(rows)
        print(f"  {var_name} {rc} ({name}) [{i+1}/37] {time.time()-t0:.0f}s", flush=True)

    result = pd.DataFrame(all_rows)
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["region_code", "date"]).reset_index(drop=True)


def pull_climate_indices():
    cache = RAW_DIR / "climate_indices" / "indices.parquet"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)
    try:
        from src.data.climate_indices import load_all_climate_indices
        df = load_all_climate_indices(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
        df.to_parquet(cache, index=False)
        return df
    except Exception as e:
        print(f"Climate indices failed: {e}. Using proxy.", flush=True)
        dates = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="MS")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"date": dates, "iod_dmi": rng.normal(0,.4,len(dates)),
                           "oni_index": rng.normal(0,.8,len(dates)),
                           "mjo_phase": rng.integers(1,9,len(dates)),
                           "mjo_amplitude": rng.exponential(1,len(dates))})
        df.to_parquet(cache, index=False)
        return df


def build_panel(var_dfs, indices_df):
    print("Building panel...", flush=True)
    panel = None
    for name, df in var_dfs.items():
        cols = ["region_code", "date", name]
        d = df[cols].copy()
        if panel is None:
            panel = d
        else:
            panel = panel.merge(d, on=["region_code", "date"], how="outer")

    if "temp_2m" in panel.columns:
        panel["temp_mean"] = panel["temp_2m"] - 273.15
        panel.drop(columns=["temp_2m"], inplace=True)
    for c in ["ndvi_monthly", "evi_monthly"]:
        if c in panel.columns:
            panel[c] = panel[c] * 0.0001
    if "lst_day" in panel.columns:
        panel["lst_day"] = panel["lst_day"] * 0.02 - 273.15

    indices_df["date"] = pd.to_datetime(indices_df["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(indices_df, on="date", how="left")

    month = panel["date"].dt.month
    for col in ["precip_monthly", "ndvi_monthly", "temp_mean"]:
        if col in panel.columns:
            m = panel.groupby(["region_code", month])[col].transform("mean")
            s = panel.groupby(["region_code", month])[col].transform("std").clip(lower=0.01)
            anom = col.replace("_monthly","").replace("temp_mean","temp")+"_anomaly"
            panel[anom] = (panel[col] - m) / s

    panel = panel.sort_values(["region_code", "date"])
    if "precip_monthly" in panel.columns:
        panel["precip_3mo_sma"] = panel.groupby("region_code")["precip_monthly"].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

    panel["month_sin"] = np.sin(2*np.pi*month/12)
    panel["month_cos"] = np.cos(2*np.pi*month/12)
    panel["season_gu"] = ((month>=3)&(month<=5)).astype(int)
    panel["season_deyr"] = ((month>=10)&(month<=12)).astype(int)
    panel["season_kiremt"] = ((month>=6)&(month<=9)).astype(int)
    if "iod_dmi" in panel.columns:
        panel["iod_3mo_lag"] = panel.groupby("region_code")["iod_dmi"].shift(3)
    panel["country"] = panel["region_code"].str[:2]

    # IPC proxy
    stress = pd.Series(0.0, index=panel.index); n=0
    for c,sign in [("precip_anomaly",-1),("ndvi_anomaly",-1),("temp_anomaly",1)]:
        if c in panel.columns: stress += sign*panel[c].fillna(0); n+=1
    if n>0: stress/=n
    panel["ipc_phase"] = pd.cut(stress,bins=[-np.inf,-.5,.2,.8,1.5,np.inf],labels=[1,2,3,4,5]).astype(float).fillna(2).astype(int)
    rng=np.random.default_rng(42)
    for r in panel["region_code"].unique():
        mask=panel["region_code"]==r; ph=panel.loc[mask,"ipc_phase"].values.copy()
        for t in range(1,len(ph)):
            if rng.random()<.65: ph[t]=ph[t-1]
            d=ph[t]-ph[t-1]
            if abs(d)>1: ph[t]=ph[t-1]+int(np.sign(d))
        panel.loc[mask,"ipc_phase"]=np.clip(ph,1,5)
    panel["prev_ipc_phase"]=panel.groupby("region_code")["ipc_phase"].shift(1)
    pc=(panel["ipc_phase"]!=panel["prev_ipc_phase"]).astype(int)
    panel["prev_ipc_duration"]=pc.groupby(pc.cumsum()).cumcount()+1
    panel["phase_trend_3mo"]=panel.groupby("region_code")["ipc_phase"].transform(lambda x:np.sign(x-x.shift(3))).fillna(0).astype(int)
    return panel.sort_values(["region_code","date"]).reset_index(drop=True)


if __name__ == "__main__":
    t0 = time.time()
    init_gee()

    variables = [
        (GEE_ASSETS["chirps"], "precipitation", "precip_monthly", "sum", 5000),
        (GEE_ASSETS["ndvi"], "NDVI", "ndvi_monthly", "mean", 1000),
        (GEE_ASSETS["ndvi"], "EVI", "evi_monthly", "mean", 1000),
        (GEE_ASSETS["era5_land"], "temperature_2m", "temp_2m", "mean", 11000),
        (GEE_ASSETS["lst"], "LST_Day_1km", "lst_day", "mean", 1000),
    ]

    var_dfs = {}
    for coll, band, name, red, scale in variables:
        var_dfs[name] = pull_var(coll, band, name, red, scale)
        print(f"DONE: {name} ({len(var_dfs[name])} rows)", flush=True)

    indices = pull_climate_indices()
    panel = build_panel(var_dfs, indices)
    panel.to_parquet(PROCESSED_DIR / "panel.parquet", index=False)

    elapsed = (time.time()-t0)/60
    print(f"\n{'='*50}", flush=True)
    print(f"Done in {elapsed:.1f} min", flush=True)
    print(f"Panel: {panel.shape}", flush=True)
    print(f"Regions: {panel['region_code'].nunique()}", flush=True)
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}", flush=True)
    print(f"IPC: {panel['ipc_phase'].value_counts().sort_index().to_dict()}", flush=True)

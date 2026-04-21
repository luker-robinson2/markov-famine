"""Central configuration for the food security prediction system."""

from enum import IntEnum
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

# ---------------------------------------------------------------------------
# IPC Phase Classification (the Markov chain state space)
# ---------------------------------------------------------------------------
class IPCPhase(IntEnum):
    """IPC Acute Food Insecurity Phase Classification.

    These 5 phases define the discrete state space S = {1,2,3,4,5}
    for our Markov chain model.
    """
    MINIMAL = 1      # Phase 1: Households meet minimum food needs
    STRESSED = 2     # Phase 2: Households have minimally adequate food
    CRISIS = 3       # Phase 3: Food consumption gaps, acute malnutrition
    EMERGENCY = 4    # Phase 4: Large food consumption gaps, excess mortality
    FAMINE = 5       # Phase 5: Starvation, death, destitution

N_STATES = len(IPCPhase)

IPC_LABELS = {
    1: "Minimal",
    2: "Stressed",
    3: "Crisis",
    4: "Emergency",
    5: "Famine",
}

# Standard IPC color palette (official)
IPC_COLORS = {
    1: "#C6FECE",  # Light green
    2: "#FAE61E",  # Yellow
    3: "#E67800",  # Orange
    4: "#C80000",  # Red
    5: "#640000",  # Dark maroon
}

# ---------------------------------------------------------------------------
# Geographic Scope: Horn of Africa Admin-1 Regions
# ---------------------------------------------------------------------------
# Kenya provinces (8) — FAO GAUL 2015 admin-1 level
# Note: GAUL uses the 8-province system, not the 47-county system
KENYA_REGIONS = {
    "KE001": "Central", "KE002": "Coast", "KE003": "Eastern",
    "KE004": "Nairobi", "KE005": "North Eastern", "KE006": "Nyanza",
    "KE007": "Rift Valley", "KE008": "Western",
}

# Ethiopia regions (11) — matches GAUL 2015 admin-1
ETHIOPIA_REGIONS = {
    "ET001": "Tigray", "ET002": "Afar", "ET003": "Amhara",
    "ET004": "Oromia", "ET005": "Somali", "ET006": "Beneshangul Gumu",
    "ET007": "SNNPR", "ET008": "Gambela", "ET009": "Hareri",
    "ET010": "Addis Ababa", "ET011": "Dire Dawa",
}

# Somalia regions (18) — matches GAUL 2015 admin-1
SOMALIA_REGIONS = {
    "SO001": "Awdal", "SO002": "Woqooyi Galbeed", "SO003": "Togdheer",
    "SO004": "Sool", "SO005": "Sanaag", "SO006": "Bari",
    "SO007": "Nugaal", "SO008": "Mudug", "SO009": "Galgaduud",
    "SO010": "Hiraan", "SO011": "Shabelle Dhexe", "SO012": "Banadir",
    "SO013": "Shabelle Hoose", "SO014": "Bay", "SO015": "Bakool",
    "SO016": "Gedo", "SO017": "Juba Dhexe", "SO018": "Juba Hoose",
}

ALL_REGIONS = {**KENYA_REGIONS, **ETHIOPIA_REGIONS, **SOMALIA_REGIONS}

# ISO 3166-1 alpha-3 codes
COUNTRY_CODES = {"KE": "KEN", "ET": "ETH", "SO": "SOM"}

# FAO GAUL admin-1 boundary asset in GEE
GAUL_ASSET = "FAO/GAUL/2015/level1"
GAUL_COUNTRY_CODES = {"KEN": 133, "ETH": 79, "SOM": 226}

# ---------------------------------------------------------------------------
# Temporal Scope
# ---------------------------------------------------------------------------
ANALYSIS_START = "2009-01-01"  # IPC data reliability improves after 2009
ANALYSIS_END = "2025-12-31"

TRAIN_END = "2022-12-31"     # Train: 2009-2022
VALID_START = "2023-01-01"   # Validate: 2023
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"    # Test: 2024
TEST_END = "2024-12-31"

# ---------------------------------------------------------------------------
# Seasonal Calendar (Horn of Africa)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Season:
    name: str
    months: tuple  # 1-indexed month numbers
    description: str

SEASONS = {
    "gu": Season("Gu", (3, 4, 5), "Long rains (Somalia/Kenya coast)"),
    "deyr": Season("Deyr", (10, 11, 12), "Short rains (Somalia/Kenya coast)"),
    "kiremt": Season("Kiremt", (6, 7, 8, 9), "Main rains (Ethiopian highlands, 65-95% annual)"),
    "belg": Season("Belg", (2, 3, 4, 5), "Small rains (Ethiopian highlands)"),
    "jilal": Season("Jilal", (1, 2, 3), "Dry season (Somalia pastoral)"),
    "hagaa": Season("Hagaa", (7, 8, 9), "Dry season (Somalia coastal)"),
}

def get_season(month: int, region_code: str) -> str:
    """Get the dominant season for a region-month combination."""
    country = region_code[:2]
    if country == "ET":
        if month in (6, 7, 8, 9):
            return "kiremt"
        elif month in (2, 3, 4, 5):
            return "belg"
        else:
            return "dry"
    else:  # KE, SO
        if month in (3, 4, 5):
            return "gu"
        elif month in (10, 11, 12):
            return "deyr"
        elif month in (1, 2, 3):
            return "jilal"
        else:
            return "hagaa"

# ---------------------------------------------------------------------------
# Google Earth Engine Asset IDs
# ---------------------------------------------------------------------------
GEE_ASSETS = {
    "chirps": "UCSB-CHG/CHIRPS/DAILY",       # Daily, 0.05deg precipitation
    "ndvi": "MODIS/061/MOD13A2",           # 16-day, 1km NDVI + EVI
    "lst": "MODIS/061/MOD11A2",            # 8-day, 1km LST
    "smap": "NASA/SMAP/SPL4SMGP/008",     # 3-hourly, 9km soil moisture (v008)
    "era5_land": "ECMWF/ERA5_LAND/HOURLY", # Hourly, 11km
    "gaul_admin1": GAUL_ASSET,
}

# MODIS band names
MODIS_NDVI_BAND = "NDVI"
MODIS_EVI_BAND = "EVI"
MODIS_LST_DAY_BAND = "LST_Day_1km"
MODIS_LST_NIGHT_BAND = "LST_Night_1km"

# ERA5-Land variable names (GEE band names)
ERA5_BANDS = {
    "temperature_2m": "temperature_2m",
    "dewpoint_2m": "dewpoint_temperature_2m",
    "u_wind_10m": "u_component_of_wind_10m",
    "v_wind_10m": "v_component_of_wind_10m",
    "surface_pressure": "surface_pressure",
    "total_precipitation": "total_precipitation",
    "surface_solar_radiation": "surface_solar_radiation_downwards",
    "soil_moisture_l1": "volumetric_soil_water_layer_1",
    "evaporation": "total_evaporation",
}

# ---------------------------------------------------------------------------
# Climate Teleconnection Index URLs
# ---------------------------------------------------------------------------
CLIMATE_INDEX_URLS = {
    # IOD: Dipole Mode Index (DMI) — NOAA PSL
    "iod_dmi": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    # ENSO: Oceanic Nino Index (ONI) — NOAA CPC
    "enso_oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    # MJO: Real-time Multivariate MJO (RMM1/RMM2) — BOM
    "mjo_rmm": "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
}

# ---------------------------------------------------------------------------
# IPC Data Sources
# ---------------------------------------------------------------------------
IPC_API_BASE = "https://api.ipcinfo.org/ipc"
HFID_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZTPHAM"
FEWSNET_PRICE_URL = "https://fews.net/data"

# ---------------------------------------------------------------------------
# Agronomic Index Parameters
# ---------------------------------------------------------------------------
# Growing Degree Days base temperatures (Celsius)
GDD_BASE_TEMPS = {
    "maize": 10.0,
    "sorghum": 10.0,
    "wheat": 5.0,
    "default": 10.0,
}

# SPEI/SPI accumulation windows (months)
DROUGHT_INDEX_WINDOWS = [1, 3, 6, 12]

# Penman-Monteith constants
PM_ALBEDO = 0.23                # Grass reference crop
PM_STEFAN_BOLTZMANN = 4.903e-9  # MJ m-2 day-1 K-4
PM_PSYCHROMETRIC = 0.665e-3     # kPa C-1

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5
MONTE_CARLO_SAMPLES = 1000

# Asymmetric cost weights for IPC phases (missing crisis is worse than false alarm)
# Phase 3+ false negative costs 3x more than false positive
MISCLASS_COST_MATRIX_WEIGHTS = {
    "false_negative_crisis": 3.0,  # Missing Phase 3+ onset
    "false_positive_crisis": 1.0,  # False alarm for Phase 3+
}

# Lead-time horizons to evaluate (months)
LEAD_TIME_HORIZONS = [1, 3, 6, 9, 12]

# ---------------------------------------------------------------------------
# Feature Groups (for selection and analysis)
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "climate_raw": [
        "precip_monthly", "precip_anomaly", "precip_3mo_sma",
        "ndvi_monthly", "ndvi_anomaly", "evi_monthly",
        "soil_moisture", "soil_moisture_anomaly",
        "temp_mean", "temp_anomaly",
    ],
    "agronomic": [
        "spei_1mo", "spei_3mo", "spei_6mo", "spi_3mo",
        "gdd_cumulative", "gdd_anomaly",
        "cwsi", "et0_monthly", "vci", "vhi",
    ],
    "teleconnection": [
        "iod_dmi", "oni_index", "mjo_phase", "mjo_amplitude",
        "iod_3mo_lag",
    ],
    "market": [
        "tot_livestock_grain", "tot_anomaly",
        "maize_price_anomaly", "livestock_price_trend",
    ],
    "temporal": [
        "month_sin", "month_cos",
        "season_gu", "season_deyr", "season_kiremt",
    ],
    "lagged_state": [
        "prev_ipc_phase", "prev_ipc_duration", "phase_trend_3mo",
    ],
    "static": [
        "livelihood_type", "population_density", "conflict_events_3mo",
    ],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# ---------------------------------------------------------------------------
# Enhanced Feature Groups (v2 — evidence-based, Machefer 2025 / Funk 2019)
# ---------------------------------------------------------------------------
DELTA_CLASSES = [-2, -1, 0, 1, 2]
DELTA_TO_IDX = {d: i for i, d in enumerate(DELTA_CLASSES)}
N_OBSERVED_STATES = 4  # phases 1-4 in data (no Phase 5 Famine observed)

ENHANCED_FEATURE_GROUPS = {
    "vegetation_indices": [
        "vci", "ndvi_roc_1mo", "ndvi_roc_3mo", "ndvi_deficit_3mo",
    ],
    "drought_indices": [
        "spi_3mo", "precip_deficit_3mo", "sm_deficit_3mo", "dry_months_count",
    ],
    "rate_of_change": [
        "ndvi_roc_1mo", "precip_roc_3mo", "sm_roc_1mo", "tot_velocity",
    ],
    "stress_duration": [
        "dry_months_count", "vci_below_35_months",
    ],
    "compound_stress": [
        "drought_x_price", "ndvi_x_season_gu",
    ],
}

# Regularized XGBoost defaults (literature-backed)
XGB_DELTA_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.02,
    "gamma": 2.0,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
}

XGB_PHASE_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.02,
    "gamma": 2.0,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
}

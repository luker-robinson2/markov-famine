"""Data acquisition modules for the food security prediction system.

Submodules
----------
cache           Parquet-based caching layer (data/raw/).
ipc_loader      IPC acute food insecurity phase data.
gee_client      Google Earth Engine climate / vegetation extraction.
climate_indices IOD, ENSO, and MJO teleconnection indices.
market_loader   FEWS NET market prices, terms of trade, and anomalies.
static_loader   ACLED conflict events, population density, livelihood zones.

Import submodules directly to avoid requiring all dependencies at package level:
    from src.data.cache import save_to_cache, load_from_cache
    from src.data.gee_client import initialize_gee, get_monthly_precipitation
"""

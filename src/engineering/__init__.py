"""Feature engineering pipeline for the food security prediction system.

Submodules
----------
spatial      Spatial alignment between naming conventions (IPC, GAUL, internal).
temporal     Temporal resampling, monthly grid alignment, cyclical encoding.
agronomic    Drought, vegetation, and evapotranspiration indices.
features     Full covariate matrix assembly.
xgb_features XGBoost-derived meta-features (SHAP, leaf indices, anomaly scores).
discretize   IPC phase validation, modal selection, gap-filling, binarization.

Import submodules directly to avoid requiring all dependencies at package level:
    from src.engineering.temporal import encode_cyclical_month
    from src.engineering.agronomic import compute_gdd
"""

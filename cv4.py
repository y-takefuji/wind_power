"""
cv4.py - Combined leakage check + VIF + leakage-free CV feature selection
================================================================================
Target: ActivePower

Stages (run simultaneously in one script):
  1. Load & clean data (missing-data policy, drop non-predictor columns)
  2. Leakage diagnostics (single-feature R2, correlation, status mapping)
  3. Leakage exclusion (concurrent-output proxies removed)
  4. VIF collinearity detection & iterative removal
  5. Nested TimeSeries CV feature selection:
       - Top-6 selection + CV6 R2 (original feature set)
       - Remove top-1 feature, re-fit ranking, select Top-5 from the re-fit
  6. Summary report saved to CSV

Feature selection methods:
  RF, XGB, RF-SHAP, XGB-SHAP, HVGS, Spearman, DEA
================================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, ttest_ind

import shap

RANDOM_STATE = 42
N_SPLITS = 5
TOP_K_FULL = 6
TOP_K_LIST = 5
MISSING_ROW_THRESHOLD = 0.10
VIF_THRESHOLD = 10
LEAK_R2_THRESHOLD = 0.95
LEAK_CORR_THRESHOLD = 0.95

TARGET = 'ActivePower'

# Concurrent-output proxies and non-predictive columns excluded
# so the model predicts power from wind and turbine-control inputs.
EXCLUDE_COLS = [
    'Unnamed: 0',
    'WTG',
    'ControlBoxTemperature',
    'GeneratorWinding1Temperature',
    'GeneratorWinding2Temperature',
    'GearboxOilTemperature',
    'GearboxBearingTemperature',
    'GeneratorRPM',
    'RotorRPM',
    'ReactivePower',
    'BearingShaftTemperature',
    'TurbineStatus',
]

# =====================================================================
# STAGE 1: Load & clean
# =====================================================================
print("=" * 70)
print("STAGE 1: Load & clean")
print("=" * 70)

df = pd.read_csv('Turbine_Data.csv')
print("Raw shape:", df.shape)

df = df.drop(columns=[c for c in ['Unnamed: 0', 'WTG'] if c in df.columns])

feature_cols = [c for c in df.columns if c != TARGET]

missing_frac = df[feature_cols].isna().mean(axis=1)
n_before = len(df)
df = df[missing_frac <= MISSING_ROW_THRESHOLD]
print(f"Removed {n_before - len(df)} rows with > {MISSING_ROW_THRESHOLD:.0%} missing features.")

n_before = len(df)
df = df.dropna(subset=[TARGET])
print(f"Removed {n_before - len(df)} rows with missing target.")
print("Shape after cleaning:", df.shape)

# =====================================================================
# STAGE 2: Leakage diagnostics (on all candidate features)
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 2: Leakage diagnostics")
print("=" * 70)

candidates = [c for c in feature_cols if c in df.columns]
X_diag = df[candidates].reset_index(drop=True)
y_diag = df[TARGET].reset_index(drop=True)

corr = X_diag.apply(lambda c: c.corr(y_diag))
print("\nPearson correlation with ActivePower:")
print(corr.sort_values(key=abs, ascending=False).round(4).to_string())

print("\nSingle-feature R2 (RF, time-ordered 80/20 split):")
split = int(len(df) * 0.8)
single_rows = []
for col in candidates:
    x = X_diag[[col]].fillna(X_diag[col].median())
    if x[col].nunique() < 2:
        single_rows.append((col, np.nan))
        continue
    m = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    m.fit(x[:split], y_diag[:split])
    single_rows.append((col, r2_score(y_diag[split:], m.predict(x[split:]))))
single_df = pd.DataFrame(single_rows, columns=['Feature', 'R2_alone']) \
              .sort_values('R2_alone', ascending=False)
print(single_df.round(4).to_string(index=False))

leak_flags = single_df[single_df['R2_alone'] > LEAK_R2_THRESHOLD]['Feature'].tolist()
if leak_flags:
    print(f"\nLeak candidates (R2_alone > {LEAK_R2_THRESHOLD}): {leak_flags}")
else:
    print(f"\nNo single feature exceeds R2_alone > {LEAK_R2_THRESHOLD}.")

if 'TurbineStatus' in df.columns:
    print("\nTurbineStatus vs ActivePower:")
    print(df.groupby('TurbineStatus')[TARGET].agg(['count', 'mean', 'std', 'min', 'max']).round(2).to_string())

# =====================================================================
# STAGE 3: Apply leakage exclusions
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 3: Apply leakage exclusions")
print("=" * 70)

excluded = [c for c in EXCLUDE_COLS if c in df.columns]
print("Excluded:", excluded)

all_nan_cols = df.columns[df.isna().all()].tolist()
feature_cols_clean = [c for c in feature_cols
                      if c not in excluded and c not in all_nan_cols]
print("Remaining features:", feature_cols_clean)

X_all = df[feature_cols_clean].reset_index(drop=True)
y = df[TARGET].reset_index(drop=True)

# =====================================================================
# STAGE 4: VIF collinearity detection & iterative removal
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 4: VIF collinearity")
print("=" * 70)

def compute_vif(X):
    X_arr = X.values.astype(float)
    vifs = {}
    for i, col in enumerate(X.columns):
        others = np.delete(X_arr, i, axis=1)
        if others.shape[1] == 0:
            vifs[col] = 1.0
            continue
        lr = LinearRegression()
        lr.fit(others, X_arr[:, i])
        r2 = lr.score(others, X_arr[:, i])
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)

def remove_collinear_iteratively(X, threshold):
    X_work = X.copy()
    removed = []
    while True:
        vif = compute_vif(X_work)
        if vif.empty or vif.iloc[0] <= threshold:
            break
        worst = vif.index[0]
        removed.append((worst, vif.iloc[0]))
        X_work = X_work.drop(columns=[worst])
    return X_work, removed

imp_full = SimpleImputer(strategy='median', keep_empty_features=True).fit(X_all)
X_all_filled = pd.DataFrame(imp_full.transform(X_all), columns=X_all.columns)

vif_initial = compute_vif(X_all_filled)
print("Initial VIF:")
print(vif_initial.round(2).to_string())

_, removed_list = remove_collinear_iteratively(X_all_filled, VIF_THRESHOLD)
print(f"\nVIF removal ({len(removed_list)} features):")
for feat, v in removed_list:
    print(f"  removed {feat:<30s} VIF={v:.2f}")

collinear_removed = [f for f, _ in removed_list]
feature_cols_final = [c for c in feature_cols_clean if c not in collinear_removed]
print(f"\nFinal feature set ({len(feature_cols_final)}): {feature_cols_final}")
print("\nVIF of final set:")
print(compute_vif(X_all_filled[feature_cols_final]).round(2).to_string())

X_full = X_all[feature_cols_final].reset_index(drop=True)

# =====================================================================
# STAGE 5: Nested TimeSeries CV feature selection (leakage-free)
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 5: Nested CV feature selection")
print("=" * 70)

def make_eval_model(method_name):
    if method_name in ['XGB', 'XGB-SHAP']:
        return XGBRegressor(random_state=RANDOM_STATE)
    return RandomForestRegressor(random_state=RANDOM_STATE)

def impute_fold(X_tr, X_te):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    imputer.fit(X_tr)
    X_tr_imp = pd.DataFrame(imputer.transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns, index=X_te.index)
    return X_tr_imp, X_te_imp

# ---- Ranking methods (all fit on TRAIN fold only) ----

def rf_feature_ranking(X, y):
    m = RandomForestRegressor(random_state=RANDOM_STATE).fit(X, y)
    return pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)

def xgb_feature_ranking(X, y):
    m = XGBRegressor(random_state=RANDOM_STATE).fit(X, y)
    return pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)

def _shap_ranking(model, X):
    X_sample = X.sample(n=min(100, len(X)), random_state=RANDOM_STATE)
    shap_values = shap.TreeExplainer(model)(X_sample)
    return pd.Series(np.abs(shap_values.values).mean(axis=0),
                     index=X.columns).sort_values(ascending=False)

def rf_shap_feature_ranking(X, y):
    m = RandomForestRegressor(random_state=RANDOM_STATE).fit(X, y)
    return _shap_ranking(m, X)

def xgb_shap_feature_ranking(X, y):
    m = XGBRegressor(random_state=RANDOM_STATE).fit(X, y)
    return _shap_ranking(m, X)

def hvgs_feature_ranking(X, y):
    return X.var(axis=0).sort_values(ascending=False)

def spearman_feature_ranking(X, y):
    corr = {c: abs(spearmanr(X[c], y)[0]) for c in X.columns}
    return pd.Series(corr).sort_values(ascending=False)

def dea_feature_ranking(X, y):
    """
    Differential Expression Analysis (DEA) feature ranking.
    The continuous target is split at its training-fold median into
    high-output and low-output groups. For each feature, a Welch t-test
    compares the two groups, and features are ranked by |t-statistic|.
    Fit on the TRAIN fold only, so no test information is used.
    """
    y_arr = np.asarray(y)
    high_mask = y_arr > np.median(y_arr)
    low_mask = ~high_mask

    scores = {}
    for c in X.columns:
        vals = X[c].values
        t_stat, _ = ttest_ind(vals[high_mask], vals[low_mask], equal_var=False)
        scores[c] = 0.0 if np.isnan(t_stat) else abs(t_stat)

    return pd.Series(scores).sort_values(ascending=False)

methods = {
    'RF': rf_feature_ranking,
    'XGB': xgb_feature_ranking,
    'RF-SHAP': rf_shap_feature_ranking,
    'XGB-SHAP': xgb_shap_feature_ranking,
    'HVGS': hvgs_feature_ranking,
    'Spearman': spearman_feature_ranking,
    'DEA': dea_feature_ranking,
}

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
results = []

for method_name, ranking_func in methods.items():
    print(f"\nMethod: {method_name}")
    cv6_scores = []
    top6_sets = []
    top5_sets = []

    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_full), start=1):
        X_tr_raw, X_te_raw = X_full.iloc[tr_idx], X_full.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        X_tr, X_te = impute_fold(X_tr_raw, X_te_raw)

        # Step A: rank on full feature set, select top-6, compute CV6 R2
        ranked = ranking_func(X_tr, y_tr)
        top6 = ranked.index[:TOP_K_FULL].tolist()
        top6_sets.append(top6)

        m = make_eval_model(method_name).fit(X_tr[top6], y_tr)
        cv6_scores.append(r2_score(y_te, m.predict(X_te[top6])))

        # Step B: remove the single highest-ranked feature, re-fit the
        # ranking on the remaining features, and select a fresh top-5
        top1 = ranked.index[0]
        X_tr_reduced = X_tr.drop(columns=[top1])
        ranked_reduced = ranking_func(X_tr_reduced, y_tr)
        top5 = ranked_reduced.index[:TOP_K_LIST].tolist()
        top5_sets.append(top5)

        print(f"  Fold {fold_idx}: R2(top{TOP_K_FULL})={cv6_scores[-1]:.3f} | "
              f"removed={top1} | top{TOP_K_LIST} after refit={top5}")

    top6_common = [f for f, _ in Counter(f for s in top6_sets for f in s).most_common(TOP_K_FULL)]
    top5_common = [f for f, _ in Counter(f for s in top5_sets for f in s).most_common(TOP_K_LIST)]

    results.append({
        'Method': method_name,
        'CV6_R2_mean': round(np.mean(cv6_scores), 3),
        'CV6_R2_std': round(np.std(cv6_scores), 3),
        'Top6_Features': ', '.join(top6_common),
        'Top5_Features': ', '.join(top5_common),
    })

# =====================================================================
# STAGE 6: Summary report
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 6: Summary")
print("=" * 70)

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
single_df.to_csv('leakage_single_feature_r2.csv', index=False)
vif_initial.rename('VIF').to_frame().to_csv('vif_initial.csv')
pd.DataFrame(removed_list, columns=['Feature', 'VIF_at_removal']).to_csv('vif_removed.csv', index=False)

print("\nSaved: result.csv, leakage_single_feature_r2.csv, vif_initial.csv, vif_removed.csv")
"""
Wind Turbine Data - Feature Selection & Cross-Validation Pipeline
====================================================================
Target: ActivePower
Feature Selection Methods: RF, XGB, RF-SHAP, XGB-SHAP, KMeans, OPTICS, HVGS, Spearman
====================================================================
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans, OPTICS
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

import shap

RANDOM_STATE = 42

# =====================================================================
# 1. Load Data
# =====================================================================
df = pd.read_csv('Turbine_Data.csv')

print("=== Before preprocessing ===")
print("Shape:", df.shape)
print(df['ActivePower'].describe())

# =====================================================================
# 2. Drop first column and 'WTG'
# =====================================================================
df = df.drop(df.columns[0], axis=1)
df = df.drop(columns=['WTG'])

# =====================================================================
# 3. Drop rows with empty cells
# =====================================================================
df = df.dropna()

print("\n=== After preprocessing ===")
print("Shape:", df.shape)
print(df['ActivePower'].describe())

# =====================================================================
# 4. Split Features / Target
# =====================================================================
target_col = 'ActivePower'
X_full = df.drop(columns=[target_col])
y = df[target_col]

feature_names_full = X_full.columns.tolist()

# =====================================================================
# Helper Function: Cross-validation with RandomForest (default settings)
# =====================================================================
def cv_r2_with_rf(X, y):
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    return scores.mean()

# =====================================================================
# Method 1: Random Forest (RF) Feature Importance
# =====================================================================
def rf_feature_ranking(X, y):
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Method 2: XGBoost (XGB) Feature Importance
# =====================================================================
def xgb_feature_ranking(X, y):
    model = XGBRegressor(random_state=RANDOM_STATE)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Method 3: RF-SHAP
# =====================================================================
def rf_shap_feature_ranking(X, y):
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X, y)

    # Use 100 randomly selected instances for SHAP (v0.48.0)
    X_sample = X.sample(n=min(100, len(X)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importances = pd.Series(mean_abs_shap, index=X.columns)
    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Method 4: XGB-SHAP
# =====================================================================
def xgb_shap_feature_ranking(X, y):
    model = XGBRegressor(random_state=RANDOM_STATE)
    model.fit(X, y)

    # Use 100 randomly selected instances for SHAP (v0.48.0)
    X_sample = X.sample(n=min(100, len(X)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importances = pd.Series(mean_abs_shap, index=X.columns)
    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Method 5: KMeans-based Feature Ranking
# (Rank features by variance of cluster centers - unsupervised)
# =====================================================================
def kmeans_feature_ranking(X, y):
    model = KMeans(random_state=RANDOM_STATE)
    model.fit(X)

    # Variance of feature values across cluster centers as importance
    centers = model.cluster_centers_
    importances = pd.Series(np.var(centers, axis=0), index=X.columns)
    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Method 6: OPTICS-based Feature Ranking
# (Rank features by variance within reachability-ordered clusters)
# =====================================================================
def optics_feature_ranking(X, y):
    model = OPTICS()
    labels = model.fit_predict(X)

    # Compute variance of feature means across clusters (excluding noise = -1)
    df_temp = X.copy()
    df_temp['cluster'] = labels
    valid = df_temp[df_temp['cluster'] != -1]

    if valid['cluster'].nunique() > 1:
        cluster_means = valid.groupby('cluster').mean()
        importances = cluster_means.var(axis=0)
    else:
        # Fallback: use overall feature variance if no valid clusters
        importances = X.var(axis=0)

    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Method 7: Highly Variable Gene Selection (HVGS)
# (Rank features by variance - adapted from single-cell genomics)
# =====================================================================
def hvgs_feature_ranking(X, y):
    importances = X.var(axis=0)
    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Method 8: Spearman Correlation
# =====================================================================
def spearman_feature_ranking(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations[col] = abs(corr)
    importances = pd.Series(correlations)
    ranked = importances.sort_values(ascending=False)
    return ranked

# =====================================================================
# Master dictionary of feature selection methods
# =====================================================================
methods = {
    'RF': rf_feature_ranking,
    'XGB': xgb_feature_ranking,
    'RF-SHAP': rf_shap_feature_ranking,
    'XGB-SHAP': xgb_shap_feature_ranking,
    'KMeans': kmeans_feature_ranking,
    'OPTICS': optics_feature_ranking,
    'HVGS': hvgs_feature_ranking,
    'Spearman': spearman_feature_ranking,
}

# Methods that have no built-in prediction -> use RF for CV
no_predict_methods = {'KMeans', 'OPTICS', 'HVGS', 'Spearman'}

# =====================================================================
# Pipeline: For EACH method independently ->
#  Step A) Fit ranking_func on FULL feature set -> rank all features
#  Step B) Select top 6 features from full ranking -> CV R^2 (top 6)
#  Step C) Remove the single highest-ranked feature from full set
#           -> creates the reduced dataset (all original features minus top-1)
#  Step D) RE-FIT the SAME ranking_func on the reduced dataset
#           -> RE-RANK features from scratch (independent re-fit)
#           -> select top 5 features from this new ranking
#  This Step D is executed for ALL 8 methods below, without exception.
# =====================================================================
results = []

for method_name, ranking_func in methods.items():
    print(f"\nProcessing method: {method_name}")

    # ---- Step A: Rank on FULL feature set (independent fit #1) ----
    ranked_full = ranking_func(X_full, y)
    top6_features = ranked_full.index[:6].tolist()

    # ---- Step B: Cross-validate with top 6 features ----
    X_top6 = X_full[top6_features]

    if method_name in no_predict_methods:
        # Use RF for CV (default settings) since these methods have no built-in predict
        cv_score = cv_r2_with_rf(X_top6, y)
    else:
        if method_name in ['RF', 'RF-SHAP']:
            model = RandomForestRegressor(random_state=RANDOM_STATE)
        else:  # XGB, XGB-SHAP
            model = XGBRegressor(random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_top6, y, cv=5, scoring='r2')
        cv_score = scores.mean()

    # ---- Step C: Remove the highest (top-1) feature -> reduced dataset ----
    top1_feature = ranked_full.index[0]
    X_reduced = X_full.drop(columns=[top1_feature])

    # ---- Step D: RE-FIT the SAME method on X_reduced (independent fit #2) ----
    # This line calls ranking_func again, from scratch, on the reduced dataset.
    # It is executed identically for RF, XGB, RF-SHAP, XGB-SHAP,
    # KMeans, OPTICS, HVGS, and Spearman — no method is skipped.
    ranked_reduced = ranking_func(X_reduced, y)
    top5_features = ranked_reduced.index[:5].tolist()

    # ---- Store result ----
    results.append({
        'Method': method_name,
        'CV6_R2': round(cv_score, 3),
        'Top6_Features': ', '.join(top6_features),
        'Top5_Features_Reduced': ', '.join(top5_features)
    })

# =====================================================================
# Summary Table
# =====================================================================
summary_df = pd.DataFrame(results)
print("\n=== Summary Table ===")
print(summary_df)

summary_df.to_csv('result.csv', index=False)
print("\nSaved summary table to 'result.csv'")
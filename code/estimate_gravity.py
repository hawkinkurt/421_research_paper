"""estimate_gravity.py"""
"""Estimates 7 models with different specifications"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")

print("=" * 70)
print("GRAVITY MODEL ESTIMATION")
print("=" * 70)
print(f"\nObservations loaded: {len(df)}")

# =============================================================================
# PREPARE VARIABLES
# =============================================================================

# Create log remoteness variables
df['log_remoteness_imp'] = np.log(df['remoteness_importer'])
df['log_remoteness_exp'] = np.log(df['remoteness_exporter'])

# Define all variables needed for estimation
base_vars = ['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
             'contig', 'comlang_off', 'col_dep_ever']
additional_vars = ['log_remoteness_imp', 'log_remoteness_exp', 'rta',
                   'efw_importer', 'efw_exporter',
                   'incentive_exporter', 'incentive_importer', 'year']

# Check variable availability
print("\n=== Variable Coverage ===")
all_vars = base_vars + additional_vars
for var in all_vars:
    if var in df.columns:
        non_missing = df[var].notna().sum()
        print(f"{var}: {non_missing} non-missing ({non_missing/len(df)*100:.1f}%)")
    else:
        print(f"{var}: NOT FOUND")

# Create analysis sample
df_est = df[all_vars].dropna().reset_index(drop=True)
print(f"\n=== Analysis Sample ===")
print(f"Complete cases: {len(df_est)} ({len(df_est)/len(df)*100:.1f}%)")

# Dependent variable
y = df_est['log_trade_real']

# Year dummies (for models with fixed effects)
year_dummies = pd.get_dummies(df_est['year'], prefix='year', drop_first=True, dtype=float)

# =============================================================================
# MODEL 1: BASELINE (GDP + Distance only)
# =============================================================================

X1 = sm.add_constant(df_est[['log_gdp_importer', 'log_gdp_exporter', 'log_dist']])
model1 = sm.OLS(y, X1).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 1: BASELINE (GDP + Distance)")
print("=" * 70)
print(model1.summary())

# =============================================================================
# MODEL 2: WITH STANDARD GRAVITY CONTROLS
# =============================================================================

X2_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
           'contig', 'comlang_off', 'col_dep_ever']
X2 = sm.add_constant(df_est[X2_vars])
model2 = sm.OLS(y, X2).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 2: WITH STANDARD CONTROLS")
print("=" * 70)
print(model2.summary())

# =============================================================================
# MODEL 3: ADDING REMOTENESS
# =============================================================================

X3_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
           'contig', 'comlang_off', 'col_dep_ever',
           'log_remoteness_imp', 'log_remoteness_exp']
X3 = sm.add_constant(df_est[X3_vars])
model3 = sm.OLS(y, X3).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 3: WITH REMOTENESS")
print("=" * 70)
print(model3.summary())

# =============================================================================
# MODEL 4: ADDING RTA
# =============================================================================

X4_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
           'contig', 'comlang_off', 'col_dep_ever',
           'log_remoteness_imp', 'log_remoteness_exp', 'rta']
X4 = sm.add_constant(df_est[X4_vars])
model4 = sm.OLS(y, X4).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 4: WITH RTA")
print("=" * 70)
print(model4.summary())

# =============================================================================
# MODEL 5: ADDING ECONOMIC FREEDOM
# =============================================================================

X5_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
           'contig', 'comlang_off', 'col_dep_ever',
           'log_remoteness_imp', 'log_remoteness_exp', 'rta',
           'efw_importer', 'efw_exporter']
X5 = sm.add_constant(df_est[X5_vars])
model5 = sm.OLS(y, X5).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 5: WITH ECONOMIC FREEDOM")
print("=" * 70)
print(model5.summary())

# =============================================================================
# MODEL 6: WITH YEAR FIXED EFFECTS
# =============================================================================

X6_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
           'contig', 'comlang_off', 'col_dep_ever',
           'log_remoteness_imp', 'log_remoteness_exp', 'rta',
           'efw_importer', 'efw_exporter']
X6 = sm.add_constant(pd.concat([df_est[X6_vars], year_dummies], axis=1))
model6 = sm.OLS(y, X6).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 6: WITH YEAR FIXED EFFECTS")
print("=" * 70)
# Print summary without year dummies cluttering output
print(f"R-squared: {model6.rsquared:.4f}")
print(f"Observations: {int(model6.nobs)}")
print("\nKey coefficients (year dummies suppressed):")
for var in X6_vars + ['const']:
    if var in model6.params:
        coef = model6.params[var]
        se = model6.bse[var]
        pval = model6.pvalues[var]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {var:<25} {coef:>10.4f} ({se:.4f}) {stars}")

# =============================================================================
# MODEL 7: FULL MODEL + FILM INCENTIVE (EXPORTER)
# =============================================================================

X7_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
           'contig', 'comlang_off', 'col_dep_ever',
           'log_remoteness_imp', 'log_remoteness_exp', 'rta',
           'efw_importer', 'efw_exporter', 'incentive_exporter']
X7 = sm.add_constant(pd.concat([df_est[X7_vars], year_dummies], axis=1))
model7 = sm.OLS(y, X7).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 7: FULL MODEL + FILM INCENTIVE (EXPORTER)")
print("=" * 70)
print(f"R-squared: {model7.rsquared:.4f}")
print(f"Observations: {int(model7.nobs)}")
print("\nKey coefficients (year dummies suppressed):")
for var in X7_vars + ['const']:
    if var in model7.params:
        coef = model7.params[var]
        se = model7.bse[var]
        pval = model7.pvalues[var]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {var:<25} {coef:>10.4f} ({se:.4f}) {stars}")

# =============================================================================
# MULTICOLLINEARITY CHECK (VIF)
# =============================================================================

print("\n" + "=" * 70)
print("VARIANCE INFLATION FACTORS (Model 7, excluding year dummies)")
print("=" * 70)

X_vif = df_est[X7_vars]
vif_data = pd.DataFrame()
vif_data['Variable'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))

if (vif_data['VIF'] > 10).any():
    print("\nWARNING: Some VIF values > 10, indicating potential multicollinearity")
else:
    print("\nNo severe multicollinearity detected (all VIF < 10)")

# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON ACROSS MODELS")
print("=" * 70)

models = [('M1', model1), ('M2', model2), ('M3', model3),
          ('M4', model4), ('M5', model5), ('M6', model6), ('M7', model7)]

key_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
            'contig', 'comlang_off', 'col_dep_ever',
            'log_remoteness_imp', 'log_remoteness_exp',
            'rta', 'efw_importer', 'efw_exporter', 'incentive_exporter']

# Print to console
header = f"{'Variable':<22}" + "".join([f"{name:>10}" for name, _ in models])
print(f"\n{header}")
print("-" * (22 + 10 * len(models)))

for var in key_vars:
    row = f"{var:<22}"
    for name, m in models:
        if var in m.params:
            coef = m.params[var]
            pval = m.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            row += f"{coef:>7.3f}{stars:<3}"
        else:
            row += f"{'--':>10}"
    print(row)

print("-" * (22 + 10 * len(models)))

row_r2 = f"{'R-squared':<22}"
row_n = f"{'Observations':<22}"
for name, m in models:
    row_r2 += f"{m.rsquared:>10.3f}"
    row_n += f"{int(m.nobs):>10}"
print(row_r2)
print(row_n)

print("\n* p<0.1, ** p<0.05, *** p<0.01")

# =============================================================================
# SAVE COMPARISON TABLE
# =============================================================================

output_dir = "C:/Users/kurtl/PycharmProjects/gravity_model/output/tables"
os.makedirs(output_dir, exist_ok=True)

# Build comparison dataframe
comparison_data = []
for var in key_vars:
    row_data = {'Variable': var}
    for name, m in models:
        if var in m.params:
            coef = m.params[var]
            se = m.bse[var]
            pval = m.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            row_data[f'{name}_coef'] = coef
            row_data[f'{name}_se'] = se
            row_data[f'{name}_stars'] = stars
        else:
            row_data[f'{name}_coef'] = np.nan
            row_data[f'{name}_se'] = np.nan
            row_data[f'{name}_stars'] = ''
    comparison_data.append(row_data)

# Add R-squared and N
comparison_data.append({
    'Variable': 'R-squared',
    **{f'{name}_coef': m.rsquared for name, m in models}
})
comparison_data.append({
    'Variable': 'Observations',
    **{f'{name}_coef': int(m.nobs) for name, m in models}
})

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)

print(f"\nComparison table saved to: {output_dir}/model_comparison.csv")

# =============================================================================
# KEY FINDINGS
# =============================================================================

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

# Distance coefficient evolution
print("\nDistance coefficient evolution:")
for name, m in models:
    if 'log_dist' in m.params:
        coef = m.params['log_dist']
        pval = m.pvalues['log_dist']
        sig = "significant" if pval < 0.1 else "not significant"
        print(f"  {name}: {coef:.3f} ({sig})")

# Incentive effect
if 'incentive_exporter' in model7.params:
    coef = model7.params['incentive_exporter']
    se = model7.bse['incentive_exporter']
    pval = model7.pvalues['incentive_exporter']
    pct_effect = (np.exp(coef) - 1) * 100
    print(f"\nFilm incentive effect (Model 7):")
    print(f"  Coefficient: {coef:.3f} (SE: {se:.3f})")
    print(f"  Percentage effect: {pct_effect:.1f}%")
    print(f"  p-value: {pval:.4f}")
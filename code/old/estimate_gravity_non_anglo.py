"""estimate_gravity_non_english.py"""
"""
Estimates gravity models EXCLUDING major English-speaking film countries:
US, UK, Canada, Australia, New Zealand

Purpose: Understand how much results are driven by the dominant 
English-speaking film production network
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Countries to exclude (major English-speaking film producers)
EXCLUDE_COUNTRIES = ['US', 'GB', 'CA', 'AU', 'NZ']

# =============================================================================
# LOAD AND FILTER DATA
# =============================================================================

df_full = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")

print("=" * 70)
print("GRAVITY MODEL ESTIMATION: EXCLUDING ENGLISH-SPEAKING COUNTRIES")
print("=" * 70)
print(f"\nExcluded countries: {', '.join(EXCLUDE_COUNTRIES)}")

print(f"\n=== FULL DATASET ===")
print(f"Total observations: {len(df_full)}")

# Filter out excluded countries (from both importer and exporter)
df = df_full[
    (~df_full['importer'].isin(EXCLUDE_COUNTRIES)) &
    (~df_full['exporter'].isin(EXCLUDE_COUNTRIES))
].copy()

print(f"\n=== FILTERED DATASET ===")
print(f"Observations after exclusion: {len(df)}")
print(f"Observations removed: {len(df_full) - len(df)} ({(len(df_full) - len(df))/len(df_full)*100:.1f}%)")

# Show what's left
print(f"\nRemaining importers: {df['importer'].nunique()}")
print(f"Remaining exporters: {df['exporter'].nunique()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

print(f"\n=== TOP 10 IMPORTERS (after exclusion) ===")
print(df['importer'].value_counts().head(10))

print(f"\n=== TOP 10 EXPORTERS (after exclusion) ===")
print(df['exporter'].value_counts().head(10))

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

if len(df_est) < 50:
    print("\nWARNING: Very small sample size. Results may be unreliable.")

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
# COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON ACROSS MODELS (Non-English Sample)")
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
# ROBUSTNESS CHECK: COUNTRY FIXED EFFECTS (Non-English Sample)
# =============================================================================

print("\n\n" + "=" * 70)
print("ROBUSTNESS CHECK: IMPORTER/EXPORTER FIXED EFFECTS (Non-English Sample)")
print("=" * 70)

# Variables that vary at the PAIR level
pair_vars = [
    'log_dist',
    'contig',
    'comlang_off',
    'col_dep_ever',
    'rta',
    'incentive_exporter'
]

# Use the filtered df and create a fresh copy for FE analysis
df_for_fe = df.dropna(subset=['log_trade_real'] + pair_vars + ['log_gdp_importer', 'log_gdp_exporter',
                                                                 'log_remoteness_imp', 'log_remoteness_exp']).copy()

print(f"Observations for FE analysis: {len(df_for_fe)}")
print(f"Unique importers: {df_for_fe['importer'].nunique()}")
print(f"Unique exporters: {df_for_fe['exporter'].nunique()}")

# Create dummies from the filtered dataframe directly
importer_dummies = pd.get_dummies(df_for_fe['importer'], prefix='imp', drop_first=True, dtype=float)
exporter_dummies = pd.get_dummies(df_for_fe['exporter'], prefix='exp', drop_first=True, dtype=float)
year_dummies_fe = pd.get_dummies(df_for_fe['year'], prefix='year', drop_first=True, dtype=float)

# Reset all indices to align
df_for_fe = df_for_fe.reset_index(drop=True)
importer_dummies = importer_dummies.reset_index(drop=True)
exporter_dummies = exporter_dummies.reset_index(drop=True)
year_dummies_fe = year_dummies_fe.reset_index(drop=True)

# --- MODEL A: Standard specification ---
print("\n" + "=" * 70)
print("MODEL A: STANDARD SPECIFICATION (GDP + Remoteness + Year FE)")
print("=" * 70)

standard_vars = [
    'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
    'contig', 'comlang_off', 'col_dep_ever',
    'log_remoteness_imp', 'log_remoteness_exp', 'rta',
    'incentive_exporter'
]

X_standard = sm.add_constant(pd.concat([
    df_for_fe[standard_vars].reset_index(drop=True),
    year_dummies_fe
], axis=1))
y_standard = df_for_fe['log_trade_real'].reset_index(drop=True)

model_standard = sm.OLS(y_standard, X_standard).fit(cov_type='HC1')

print(f"R-squared: {model_standard.rsquared:.4f}")
print(f"Observations: {int(model_standard.nobs)}")
print("\nKey coefficients (year dummies suppressed):")
for var in standard_vars:
    if var in model_standard.params:
        coef = model_standard.params[var]
        se = model_standard.bse[var]
        pval = model_standard.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {var:<25} {coef:>10.4f} ({se:.4f}) {stars}")

# --- MODEL B: Fixed Effects ---
print("\n" + "=" * 70)
print("MODEL B: IMPORTER + EXPORTER + YEAR FIXED EFFECTS")
print("=" * 70)

X_fe = sm.add_constant(pd.concat([
    df_for_fe[pair_vars].reset_index(drop=True),
    importer_dummies,
    exporter_dummies,
    year_dummies_fe
], axis=1))
y_fe = df_for_fe['log_trade_real'].reset_index(drop=True)

model_fe = sm.OLS(y_fe, X_fe).fit(cov_type='HC1')

print(f"R-squared: {model_fe.rsquared:.4f}")
print(f"Observations: {int(model_fe.nobs)}")
print(f"Number of parameters: {len(model_fe.params)}")
print("\nKey coefficients (country and year dummies suppressed):")
for var in pair_vars:
    if var in model_fe.params:
        coef = model_fe.params[var]
        se = model_fe.bse[var]
        pval = model_fe.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {var:<25} {coef:>10.4f} ({se:.4f}) {stars}")

# --- Comparison ---
print("\n" + "=" * 70)
print("COMPARISON: STANDARD vs FIXED EFFECTS (Non-English Sample)")
print("=" * 70)

print(f"\n{'Variable':<25} {'Standard':>12} {'Fixed Effects':>14} {'Difference':>12}")
print("-" * 65)

for var in pair_vars:
    coef_std = model_standard.params.get(var, np.nan)
    coef_fe = model_fe.params.get(var, np.nan)

    if pd.notna(coef_std) and pd.notna(coef_fe):
        diff = coef_fe - coef_std

        pval_std = model_standard.pvalues.get(var, 1)
        stars_std = '***' if pval_std < 0.01 else '**' if pval_std < 0.05 else '*' if pval_std < 0.1 else ''

        pval_fe = model_fe.pvalues.get(var, 1)
        stars_fe = '***' if pval_fe < 0.01 else '**' if pval_fe < 0.05 else '*' if pval_fe < 0.1 else ''

        print(f"{var:<25} {coef_std:>9.3f}{stars_std:<3} {coef_fe:>11.3f}{stars_fe:<3} {diff:>+10.3f}")

print("-" * 65)
print(f"{'R-squared':<25} {model_standard.rsquared:>12.3f} {model_fe.rsquared:>14.3f}")
print(f"{'Observations':<25} {int(model_standard.nobs):>12} {int(model_fe.nobs):>14}")

# =============================================================================
# KEY FINDINGS COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("KEY FINDINGS: INCENTIVE EFFECT (Non-English Sample)")
print("=" * 70)

print(f"""
NON-ENGLISH SAMPLE RESULTS:

Standard specification (with year FE):
  Coefficient: {model_standard.params.get('incentive_exporter', np.nan):.4f}
  SE: {model_standard.bse.get('incentive_exporter', np.nan):.4f}
  p-value: {model_standard.pvalues.get('incentive_exporter', np.nan):.4f}
  Percentage effect: {(np.exp(model_standard.params.get('incentive_exporter', 0)) - 1) * 100:.1f}%

Fixed effects specification:
  Coefficient: {model_fe.params.get('incentive_exporter', np.nan):.4f}
  SE: {model_fe.bse.get('incentive_exporter', np.nan):.4f}
  p-value: {model_fe.pvalues.get('incentive_exporter', np.nan):.4f}
  Percentage effect: {(np.exp(model_fe.params.get('incentive_exporter', 0)) - 1) * 100:.1f}%
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: FULL SAMPLE vs NON-ENGLISH SAMPLE")
print("=" * 70)

print(f"""
This analysis excludes: {', '.join(EXCLUDE_COUNTRIES)}

Sample size impact:
  Full sample: {len(df_full)} observations
  Non-English sample: {len(df_est)} observations ({len(df_est)/len(df_full)*100:.1f}% of original)

Key questions answered:
1. Does distance behave normally without English-speaking countries?
2. Does the incentive effect hold in non-English film markets?
3. Does the incentive effect survive fixed effects in this subsample?

Compare these results to the full-sample analysis to understand
how much the English-speaking film production network drives the findings.
""")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_dir = "C:/Users/kurtl/PycharmProjects/gravity_model/output/tables"
os.makedirs(output_dir, exist_ok=True)

# Build comparison dataframe
comparison_data = []
for var in key_vars:
    row_data = {'Variable': var}
    for name, m in models:
        if var in m.params:
            row_data[f'{name}_coef'] = m.params[var]
            row_data[f'{name}_se'] = m.bse[var]
            row_data[f'{name}_pval'] = m.pvalues[var]
        else:
            row_data[f'{name}_coef'] = np.nan
    comparison_data.append(row_data)

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(f"{output_dir}/model_comparison_non_english.csv", index=False)

print(f"\nResults saved to: {output_dir}/model_comparison_non_english.csv")
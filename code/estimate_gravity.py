"""
estimate_gravity.py
Updated to include RTA, Economic Freedom, and Remoteness variables
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")

print("=" * 70)
print("GRAVITY MODEL ESTIMATION - EXPANDED SPECIFICATION")
print("=" * 70)
print(f"\nObservations loaded: {len(df)}")

# =============================================================================
# CHECK AVAILABLE VARIABLES
# =============================================================================

print("\n=== Available Variables ===")
print(df.columns.tolist())

# Check for new variables
new_vars = ['rta', 'efw_importer', 'efw_exporter', 'remoteness_importer', 'remoteness_exporter', 'incentive_exporter', 'incentive_importer']
print("\n=== New Variable Coverage ===")
for var in new_vars:
    if var in df.columns:
        non_missing = df[var].notna().sum()
        print(f"{var}: {non_missing} non-missing ({non_missing/len(df)*100:.1f}%)")
    else:
        print(f"{var}: NOT FOUND IN DATASET")

# =============================================================================
# PREPARE ANALYSIS SAMPLE
# =============================================================================

# Define all variables needed
base_vars = ['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
             'contig', 'comlang_off', 'col_dep_ever']
new_vars_available = [v for v in new_vars if v in df.columns]

all_vars = base_vars + new_vars_available + ['year']
df_est = df[all_vars].dropna()

print(f"\n=== Analysis Sample ===")
print(f"Complete cases: {len(df_est)} ({len(df_est)/len(df)*100:.1f}% of loaded data)")

# =============================================================================
# MODEL 1: BASELINE (GDP + Distance only)
# =============================================================================

y = df_est['log_trade_real']
X1 = df_est[['log_gdp_importer', 'log_gdp_exporter', 'log_dist']]
X1 = sm.add_constant(X1)

model1 = sm.OLS(y, X1).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 1: BASELINE (GDP + Distance)")
print("=" * 70)
print(model1.summary())

# =============================================================================
# MODEL 2: WITH STANDARD GRAVITY CONTROLS
# =============================================================================

X2 = df_est[['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
             'contig', 'comlang_off', 'col_dep_ever']]
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 2: WITH STANDARD CONTROLS")
print("=" * 70)
print(model2.summary())

# =============================================================================
# MODEL 3: ADDING REMOTENESS
# =============================================================================

if 'remoteness_importer' in df_est.columns and 'remoteness_exporter' in df_est.columns:
    # Create log remoteness
    df_est['log_remoteness_imp'] = np.log(df_est['remoteness_importer'])
    df_est['log_remoteness_exp'] = np.log(df_est['remoteness_exporter'])

    X3 = df_est[['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
                 'contig', 'comlang_off', 'col_dep_ever',
                 'log_remoteness_imp', 'log_remoteness_exp']]
    X3 = sm.add_constant(X3)

    model3 = sm.OLS(y, X3).fit(cov_type='HC1')

    print("\n" + "=" * 70)
    print("MODEL 3: WITH REMOTENESS")
    print("=" * 70)
    print(model3.summary())
else:
    print("\n*** Remoteness variables not available - skipping Model 3 ***")
    model3 = None

# =============================================================================
# MODEL 4: ADDING RTA
# =============================================================================

if 'rta' in df_est.columns:
    X4_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
               'contig', 'comlang_off', 'col_dep_ever', 'rta']
    if 'log_remoteness_imp' in df_est.columns:
        X4_vars += ['log_remoteness_imp', 'log_remoteness_exp']

    X4 = df_est[X4_vars]
    X4 = sm.add_constant(X4)

    model4 = sm.OLS(y, X4).fit(cov_type='HC1')

    print("\n" + "=" * 70)
    print("MODEL 4: WITH RTA")
    print("=" * 70)
    print(model4.summary())
else:
    print("\n*** RTA variable not available - skipping Model 4 ***")
    model4 = None

# =============================================================================
# MODEL 5: ADDING ECONOMIC FREEDOM
# =============================================================================

if 'efw_importer' in df_est.columns and 'efw_exporter' in df_est.columns:
    X5_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
               'contig', 'comlang_off', 'col_dep_ever']
    if 'log_remoteness_imp' in df_est.columns:
        X5_vars += ['log_remoteness_imp', 'log_remoteness_exp']
    if 'rta' in df_est.columns:
        X5_vars += ['rta']
    X5_vars += ['efw_importer', 'efw_exporter']

    X5 = df_est[X5_vars]
    X5 = sm.add_constant(X5)

    model5 = sm.OLS(y, X5).fit(cov_type='HC1')

    print("\n" + "=" * 70)
    print("MODEL 5: FULL MODEL (with EFW)")
    print("=" * 70)
    print(model5.summary())
else:
    print("\n*** EFW variables not available - skipping Model 5 ***")
    model5 = None

# =============================================================================
# MODEL 6: FULL MODEL WITH YEAR FIXED EFFECTS
# =============================================================================

# Reset index to avoid alignment issues
df_est_reset = df_est.reset_index(drop=True)
y6 = df_est_reset['log_trade_real']

year_dummies = pd.get_dummies(df_est_reset['year'], prefix='year', drop_first=True, dtype=float)

X6_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
           'contig', 'comlang_off', 'col_dep_ever']
if 'log_remoteness_imp' in df_est_reset.columns:
    X6_vars += ['log_remoteness_imp', 'log_remoteness_exp']
if 'rta' in df_est_reset.columns:
    X6_vars += ['rta']
if 'efw_importer' in df_est_reset.columns:
    X6_vars += ['efw_importer', 'efw_exporter']

X6 = pd.concat([df_est_reset[X6_vars], year_dummies], axis=1)
X6 = sm.add_constant(X6)

model6 = sm.OLS(y6, X6).fit(cov_type='HC1')

print("\n" + "=" * 70)
print("MODEL 6: FULL MODEL WITH YEAR FIXED EFFECTS")
print("=" * 70)
print(model6.summary())

# =============================================================================
# MODEL 7: FULL MODEL + INCENTIVE
# =============================================================================

if 'incentive_exporter' in df.columns:
    df_est_reset['incentive_exporter'] = df.loc[df_est.index, 'incentive_exporter'].values

    X7_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
               'contig', 'comlang_off', 'col_dep_ever',
               'log_remoteness_imp', 'log_remoteness_exp',
               'rta', 'efw_importer', 'efw_exporter', 'incentive_exporter']

    X7 = pd.concat([df_est_reset[X7_vars], year_dummies], axis=1)
    X7 = sm.add_constant(X7)

    model7 = sm.OLS(y6, X7).fit(cov_type='HC1')

    print("\n" + "=" * 70)
    print("MODEL 7: FULL MODEL + FILM INCENTIVE")
    print("=" * 70)
    print(model7.summary())
else:
    model7 = None

# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON ACROSS MODELS")
print("=" * 70)

key_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist',
            'contig', 'comlang_off', 'col_dep_ever',
            'log_remoteness_imp', 'log_remoteness_exp',
            'rta', 'efw_importer', 'efw_exporter', 'incentive_exporter']

models = [('M1', model1), ('M2', model2), ('M3', model3),
          ('M4', model4), ('M5', model5), ('M6', model6), ('M7', model7)]
models = [(name, m) for name, m in models if m is not None]

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
print("\nKey question: Does the distance coefficient become negative in any specification?")
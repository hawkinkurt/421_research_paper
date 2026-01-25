"""estimate_gravity.py"""
import pandas as pd
import statsmodels.formula.api as smf

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")

print(f"=== Analysis Dataset ===")
print(f"Observations: {len(df)}")
print(f"Unique importer-exporter pairs: {df.groupby(['importer', 'exporter']).ngroups}")
print(f"Year range: {df['year'].min()} to {df['year'].max()}")

# =============================================================================
# MODEL 1: BASIC GRAVITY (OLS)
# =============================================================================

print("\n" + "="*70)
print("MODEL 1: Basic Gravity Model (OLS)")
print("="*70)

model1 = smf.ols(
    'log_trade_real ~ log_gdp_importer + log_gdp_exporter + log_dist',
    data=df
).fit()

print(model1.summary())

# =============================================================================
# MODEL 2: ADD GRAVITY CONTROLS
# =============================================================================

print("\n" + "="*70)
print("MODEL 2: Gravity with Controls (OLS)")
print("="*70)

model2 = smf.ols(
    'log_trade_real ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + colony',
    data=df
).fit()

print(model2.summary())

# =============================================================================
# MODEL 3: ADD YEAR FIXED EFFECTS
# =============================================================================

print("\n" + "="*70)
print("MODEL 3: Gravity with Year Fixed Effects")
print("="*70)

model3 = smf.ols(
    'log_trade_real ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + colony + C(year)',
    data=df
).fit()

print(model3.summary())

# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n" + "="*70)
print("COEFFICIENT COMPARISON")
print("="*70)

key_vars = ['log_gdp_importer', 'log_gdp_exporter', 'log_dist', 'contig', 'comlang_off', 'colony']

print(f"\n{'Variable':<20} {'Model 1':>12} {'Model 2':>12} {'Model 3':>12}")
print("-" * 58)

for var in key_vars:
    m1 = f"{model1.params.get(var, 'N/A'):>12.3f}" if var in model1.params else f"{'N/A':>12}"
    m2 = f"{model2.params.get(var, 'N/A'):>12.3f}" if var in model2.params else f"{'N/A':>12}"
    m3 = f"{model3.params.get(var, 'N/A'):>12.3f}" if var in model3.params else f"{'N/A':>12}"
    print(f"{var:<20} {m1} {m2} {m3}")

print("-" * 58)
print(f"{'R-squared':<20} {model1.rsquared:>12.3f} {model2.rsquared:>12.3f} {model3.rsquared:>12.3f}")
print(f"{'Observations':<20} {int(model1.nobs):>12} {int(model2.nobs):>12} {int(model3.nobs):>12}")
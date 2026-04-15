"""
patch_efwresid.py
=================
Run this script to patch data_fetch_clean.py with EFWRESID blocks.

Usage:
    python patch_efwresid.py

Expects data_fetch_clean.py in the same directory.
Creates data_fetch_clean.py.bak as backup, then modifies in place.
"""

import os
import shutil

FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_fetch_clean.py')

# ─── EFWRESID block for Phase 7 (insert before "# --- 7f:") ───
EFWRESID_BLOCK_7 = '''
    # --- 7e2: Create income-adjusted EFW (EFWRESID) ---
    print("\\n--- 7e2: Create income-adjusted EFW (EFWRESID, pooled OLS) ---")

    import statsmodels.api as sm_ols

    # Pooled regression: EFW = alpha + beta * ln(GDP) + epsilon
    # Residual = EFW component orthogonal to income (Kimura & Lee, 2006)
    # Exporter side
    efw_gdp_exp = merged[['efw_exporter', 'log_gdp_exporter']].dropna()
    if len(efw_gdp_exp) > 50:
        X_exp = sm_ols.add_constant(efw_gdp_exp['log_gdp_exporter'])
        res_exp = sm_ols.OLS(efw_gdp_exp['efw_exporter'], X_exp).fit()
        merged.loc[efw_gdp_exp.index, 'efwresid_exporter'] = res_exp.resid
        print(f"  Exporter EFWRESID: N={int(res_exp.nobs)}, "
              f"R2={res_exp.rsquared:.4f}, beta(ln GDP)={res_exp.params.iloc[1]:.4f}")
    else:
        merged['efwresid_exporter'] = np.nan
        print("  WARNING: too few obs for exporter EFWRESID regression")

    # Importer side
    efw_gdp_imp = merged[['efw_importer', 'log_gdp_importer']].dropna()
    if len(efw_gdp_imp) > 50:
        X_imp = sm_ols.add_constant(efw_gdp_imp['log_gdp_importer'])
        res_imp = sm_ols.OLS(efw_gdp_imp['efw_importer'], X_imp).fit()
        merged.loc[efw_gdp_imp.index, 'efwresid_importer'] = res_imp.resid
        print(f"  Importer EFWRESID: N={int(res_imp.nobs)}, "
              f"R2={res_imp.rsquared:.4f}, beta(ln GDP)={res_imp.params.iloc[1]:.4f}")
    else:
        merged['efwresid_importer'] = np.nan
        print("  WARNING: too few obs for importer EFWRESID regression")

    n_resid = merged['efwresid_exporter'].notna().sum()
    print(f"  EFWRESID coverage: {n_resid:,} / {len(merged):,} "
          f"({n_resid/len(merged)*100:.1f}%)")

'''

# ─── EFWRESID block for Phase 8 (insert before "# --- 8e:") ───
EFWRESID_BLOCK_8 = '''
    # --- 8d2: Create income-adjusted EFW (EFWRESID) ---
    print("\\n--- 8d2: Create income-adjusted EFW (EFWRESID, pooled OLS) ---")

    import statsmodels.api as sm_ols

    # Pooled regression: EFW = alpha + beta * ln(GDP) + epsilon
    # Residual = EFW component orthogonal to income (Kimura & Lee, 2006)
    # Exporter side
    efw_gdp_exp = merged[['efw_exporter', 'log_gdp_exporter']].dropna()
    if len(efw_gdp_exp) > 50:
        X_exp = sm_ols.add_constant(efw_gdp_exp['log_gdp_exporter'])
        res_exp = sm_ols.OLS(efw_gdp_exp['efw_exporter'], X_exp).fit()
        merged.loc[efw_gdp_exp.index, 'efwresid_exporter'] = res_exp.resid
        print(f"  Exporter EFWRESID: N={int(res_exp.nobs)}, "
              f"R2={res_exp.rsquared:.4f}, beta(ln GDP)={res_exp.params.iloc[1]:.4f}")
    else:
        merged['efwresid_exporter'] = np.nan
        print("  WARNING: too few obs for exporter EFWRESID regression")

    # Importer side
    efw_gdp_imp = merged[['efw_importer', 'log_gdp_importer']].dropna()
    if len(efw_gdp_imp) > 50:
        X_imp = sm_ols.add_constant(efw_gdp_imp['log_gdp_importer'])
        res_imp = sm_ols.OLS(efw_gdp_imp['efw_importer'], X_imp).fit()
        merged.loc[efw_gdp_imp.index, 'efwresid_importer'] = res_imp.resid
        print(f"  Importer EFWRESID: N={int(res_imp.nobs)}, "
              f"R2={res_imp.rsquared:.4f}, beta(ln GDP)={res_imp.params.iloc[1]:.4f}")
    else:
        merged['efwresid_importer'] = np.nan
        print("  WARNING: too few obs for importer EFWRESID regression")

    n_resid = merged['efwresid_exporter'].notna().sum()
    print(f"  EFWRESID coverage: {n_resid:,} / {len(merged):,} "
          f"({n_resid/len(merged)*100:.1f}%)")

'''

# ─── Anchors: insert BEFORE these lines ───
ANCHOR_7F = "    # --- 7f: Create incentive variables ---"
ANCHOR_8E = "    # --- 8e: Create incentive variables ---"


def main():
    if not os.path.exists(FILENAME):
        print(f"ERROR: {FILENAME} not found")
        return

    # Backup
    backup = FILENAME + '.bak'
    shutil.copy2(FILENAME, backup)
    print(f"Backup created: {backup}")

    with open(FILENAME, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check anchors exist
    if ANCHOR_7F not in content:
        print(f"ERROR: Could not find anchor for Phase 7: '{ANCHOR_7F}'")
        return
    if ANCHOR_8E not in content:
        print(f"ERROR: Could not find anchor for Phase 8: '{ANCHOR_8E}'")
        return

    # Check not already patched
    if 'efwresid_exporter' in content:
        print("WARNING: EFWRESID already appears in file — skipping to avoid double-patch")
        return

    # Insert Phase 7 block (before 7f anchor)
    content = content.replace(
        ANCHOR_7F,
        EFWRESID_BLOCK_7 + ANCHOR_7F
    )

    # Insert Phase 8 block (before 8e anchor)
    content = content.replace(
        ANCHOR_8E,
        EFWRESID_BLOCK_8 + ANCHOR_8E
    )

    with open(FILENAME, 'w', encoding='utf-8') as f:
        f.write(content)

    # Verify
    with open(FILENAME, 'r', encoding='utf-8') as f:
        patched = f.read()

    n_efwresid = patched.count('efwresid_exporter')
    print(f"\nPatched successfully!")
    print(f"  'efwresid_exporter' appears {n_efwresid} times in patched file")
    print(f"  Phase 7 block inserted before: {ANCHOR_7F}")
    print(f"  Phase 8 block inserted before: {ANCHOR_8E}")

    # Syntax check
    try:
        compile(patched, FILENAME, 'exec')
        print("  Syntax check: PASSED")
    except SyntaxError as e:
        print(f"  Syntax check: FAILED — {e}")


if __name__ == '__main__':
    main()
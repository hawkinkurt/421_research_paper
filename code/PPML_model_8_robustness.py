"""
m8_ppml_robustness.py
=====================
Standalone robustness check for Model 8 (country-pair fixed effects).

Re-estimates Model 8 by Poisson Pseudo-Maximum Likelihood with country-pair
and year fixed effects, addressing the Chen and Roth (2024, QJE) critique of
ln(y+1) transformations when zeros are common. Runs in parallel to the main
imdb_filming_gravity_analysis.py script — uses the same data files, the same
incentive intro dates, and the same Anglo subsample definitions, so the
estimates are directly comparable to Model 8 OLS results in Table 8.

Estimator: PPML with high-dimensional fixed effects via pyfixest.fepois,
implementing the Correia-Guimaraes-Zylkin (2020) algorithm. Standard errors
clustered at the country-pair level.

Specifications run (film count outcome only):
  1. Full sample
  2. Excluding Anglo-to-Anglo pairs
  3. Excluding all Anglo pairs

Outputs:
  - Console comparison vs. OLS Model 8 destination incentive coefficient
  - CSV file with all coefficients side-by-side at
    {PROJECT_DIR}\\results\\ppml_robustness\\m8_ppml_robustness_fc.csv
  - LaTeX table at
    {PROJECT_DIR}\\results\\ppml_robustness\\m8_ppml_robustness_fc.tex

Notes
-----
- Budgets and Models 9/10 are NOT re-estimated; the log transformation in
  those specifications is acknowledged in the limitations section instead.
- PPML drops pairs that are zero in every year (uninformative under pair FE)
  and further observations identified by the FE separation check. Effective
  sample sizes therefore differ from the OLS specifications.

Author: Kurt Hawkins
Project: 421_research_paper / gravity_model
Run AFTER imdb_filming_gravity_analysis.py has produced its console output,
or independently if the data files exist.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    import pyfixest as pf
except ImportError:
    print("ERROR: pyfixest is required for this robustness check.")
    print("       Install with: pip install pyfixest")
    sys.exit(1)


# =============================================================================
# CONFIGURATION — mirrors imdb_filming_gravity_analysis.py
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

ANALYSIS_FILE = f'{PROJECT_DIR}\\data\\processed\\imdb\\imdb_gravity_analysis.csv'
INCENTIVE_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\film_incentive_intro_dates.csv'

# Output directory for the LaTeX/CSV artefacts
RESULTS_DIR = f'{PROJECT_DIR}\\results\\ppml_robustness'

ANGLO = {'US', 'GB', 'CA', 'AU', 'NZ'}

# Time-varying RHS variables for Model 8, matching TV_FULL in the main script.
# Pair FE absorbs all time-invariant bilateral controls (distance, contiguity,
# common language, colonial relationship), so they drop out automatically.
TV_FULL = [
    'log_gdp_importer', 'log_gdp_exporter',
    'remoteness_importer', 'remoteness_exporter',
    'efw_exporter', 'efw_importer',
    'rta', 'incentive_exporter', 'incentive_importer',
]

# Dependent variable: PPML uses raw counts, not the log transform
DEP_VAR = 'num_films'

# pyfixest convergence settings
FEPOIS_KWARGS = dict(
    iwls_tol=1e-7,
    iwls_maxiter=100,
    separation_check=['fe'],
)

# OLS Model 8 destination incentive coefficients, taken directly from Table 8
# of the paper, used here for the side-by-side console comparison only.
OLS_M8_DEST_COEF = {
    'Full sample':         0.0907,
    'Excl Anglo-to-Anglo': 0.0940,
    'Excl all Anglo':      0.0717,
}


# =============================================================================
# DATA LOADING & PREP
# =============================================================================

def load_panel() -> pd.DataFrame:
    """Load the analysis panel produced by data_fetch_clean.py Phase 8."""
    if not os.path.exists(ANALYSIS_FILE):
        print(f"ERROR: Could not find analysis file at {ANALYSIS_FILE}")
        print("       Run imdb_filming_gravity_analysis.py first to generate it.")
        sys.exit(1)

    df = pd.read_csv(ANALYSIS_FILE)
    print(f"Loaded analysis panel: {len(df):,} rows, "
          f"{df['exporter'].nunique()} exporters, "
          f"{df['importer'].nunique()} importers, "
          f"years {df['year'].min()}-{df['year'].max()}")

    # Validate required columns
    required = set(TV_FULL) | {DEP_VAR, 'exporter', 'importer', 'year'}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Panel missing required columns: {missing}")
        sys.exit(1)

    return df


def prepare_for_ppml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the panel for PPML pair FE estimation.

    Steps:
      1. Construct pair_id matching the main script's convention.
      2. Drop pairs that are zero in every year — uninformative under pair FE
         and frequently cause convergence issues for the iterative reweighting.
      3. Drop rows with missing covariates (PPML cannot handle NaN).
    """
    df = df.copy()
    df['pair_id'] = df['exporter'] + '_' + df['importer']

    # Drop pairs with all-zero outcomes
    pair_max = df.groupby('pair_id')[DEP_VAR].transform('max')
    n_before = len(df)
    df = df.loc[pair_max > 0].copy()
    n_dropped_zero = n_before - len(df)
    print(f"Dropped {n_dropped_zero:,} observations from pairs with all-zero outcomes")

    # Drop rows with missing covariates
    n_before = len(df)
    df = df.dropna(subset=TV_FULL + [DEP_VAR]).copy()
    n_dropped_na = n_before - len(df)
    if n_dropped_na > 0:
        print(f"Dropped {n_dropped_na:,} observations with missing covariates")

    print(f"Panel ready for estimation: {len(df):,} rows, "
          f"{df['pair_id'].nunique():,} pairs")

    return df


def make_subsample(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Mirror the subsample logic from imdb_filming_gravity_analysis.py."""
    if kind == 'Full sample':
        return df
    if kind == 'Excl Anglo-to-Anglo':
        # Drop pairs where BOTH origin and destination are Anglo
        mask = ~(df['exporter'].isin(ANGLO) & df['importer'].isin(ANGLO))
        return df.loc[mask].copy()
    if kind == 'Excl all Anglo':
        # Drop pairs where EITHER side is Anglo
        mask = ~(df['exporter'].isin(ANGLO) | df['importer'].isin(ANGLO))
        return df.loc[mask].copy()
    raise ValueError(f"Unknown subsample: {kind}")


# =============================================================================
# ESTIMATION
# =============================================================================

def estimate(df_sub: pd.DataFrame, label: str) -> dict | None:
    """Estimate PPML pair FE on a subsample. Returns None on failure."""
    print(f"\n{'='*70}")
    print(f"PPML PAIR FE — {label}")
    print(f"{'='*70}")
    print(f"  Observations: {len(df_sub):,}")
    print(f"  Pairs:        {df_sub['pair_id'].nunique():,}")
    print(f"  Years:        {df_sub['year'].nunique()}")
    print(f"  Share zeros:  {(df_sub[DEP_VAR] == 0).mean():.1%}")

    if len(df_sub) < 50 or df_sub['pair_id'].nunique() < 10:
        print(f"  SKIPPED — too few observations or pairs")
        return None

    formula = f"{DEP_VAR} ~ " + " + ".join(TV_FULL) + " | pair_id + year"
    print(f"  Formula: {formula}")

    try:
        model = pf.fepois(
            formula,
            data=df_sub,
            vcov={'CRV1': 'pair_id'},
            **FEPOIS_KWARGS,
        )
    except Exception as e:
        print(f"  ESTIMATION FAILED: {type(e).__name__}: {e}")
        return None

    coef = model.coef()
    se = model.se()
    pval = model.pvalue()

    n_used = int(model._N) if hasattr(model, '_N') else len(df_sub)
    print(f"  Effective N:  {n_used:,} (separation drops: {len(df_sub) - n_used:,})")

    # Print coefficients
    print(f"\n  {'Variable':<26} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5} {'% effect':>10}")
    print('  ' + '-' * 73)
    for var in TV_FULL:
        if var in coef.index:
            c = coef[var]
            s = se[var]
            p = pval[var]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            pct = (np.exp(c) - 1) * 100
            print(f"  {var:<26} {c:>10.4f} {s:>10.4f} {p:>8.4f} {stars:>5} {pct:>+9.1f}%")
        else:
            print(f"  {var:<26} {'--':>10}")

    return {
        'label': label,
        'model': model,
        'coef': coef,
        'se': se,
        'pval': pval,
        'n_used': n_used,
        'n_raw': len(df_sub),
        'pairs': df_sub['pair_id'].nunique(),
    }


# =============================================================================
# OUTPUT
# =============================================================================

def write_csv(results: dict[str, dict | None], path: str) -> None:
    """Write a CSV with all coefficients side-by-side."""
    rows = []
    for var in TV_FULL:
        row = {'Variable': var}
        for label, r in results.items():
            if r is None:
                row[label] = '—'
                continue
            if var in r['coef'].index:
                c = r['coef'][var]
                s = r['se'][var]
                p = r['pval'][var]
                stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
                row[label] = f"{c:.4f}{stars} (SE {s:.4f})"
            else:
                row[label] = '—'
        rows.append(row)

    # Diagnostic rows
    rows.append({'Variable': '— DIAGNOSTICS —', **{k: '' for k in results}})
    rows.append({'Variable': 'N (raw)',
                 **{k: f"{r['n_raw']:,}" if r else '—' for k, r in results.items()}})
    rows.append({'Variable': 'N (used)',
                 **{k: f"{r['n_used']:,}" if r else '—' for k, r in results.items()}})
    rows.append({'Variable': 'Pairs',
                 **{k: f"{r['pairs']:,}" if r else '—' for k, r in results.items()}})

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nCSV written to: {path}")


def write_latex(results: dict[str, dict | None], path: str) -> None:
    """Write a LaTeX table matching the style of Table 8 in the paper."""
    pretty = {
        'log_gdp_importer':   r'$\ln$ GDP (origin)',
        'log_gdp_exporter':   r'$\ln$ GDP (destination)',
        'remoteness_importer': 'Remoteness (origin)',
        'remoteness_exporter': 'Remoteness (destination)',
        'efw_importer':        'Econ. freedom (origin)',
        'efw_exporter':        'Econ. freedom (destination)',
        'rta':                 'RTA',
        'incentive_exporter':  'Incentive (destination)',
        'incentive_importer':  'Incentive (origin)',
    }
    short_labels = {
        'Full sample':         'Full',
        'Excl Anglo-to-Anglo': 'Excl A-to-A',
        'Excl all Anglo':      'Excl Anglo',
    }
    labels = list(results.keys())
    n_cols = len(labels)

    lines = [
        r'\begin{table}[ht]',
        r'\centering\small',
        r'\caption{Model 8 Robustness: PPML Pair Fixed Effects --- IMDb Film Counts}',
        r'\label{tab:m8_ppml_robustness}',
        r'\begin{tabular}{l' + 'c' * n_cols + '}',
        r'\hline\hline',
        ' & ' + ' & '.join([short_labels.get(l, l) for l in labels]) + r' \\',
        r'\hline',
    ]

    for var in TV_FULL:
        coef_cells = []
        se_cells = []
        for label in labels:
            r = results[label]
            if r is None or var not in r['coef'].index:
                coef_cells.append('---')
                se_cells.append('')
                continue
            c = r['coef'][var]
            s = r['se'][var]
            p = r['pval'][var]
            stars = '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.10 else ''
            coef_cells.append(f'${c:.4f}{stars}$')
            se_cells.append(f'$({s:.4f})$')
        lines.append(f"{pretty.get(var, var)} & " + ' & '.join(coef_cells) + r' \\')
        lines.append(' & ' + ' & '.join(se_cells) + r' \\')

    lines.append(r'\hline')
    lines.append('Pair FE & ' + ' & '.join(['Yes'] * n_cols) + r' \\')
    lines.append('Year FE & ' + ' & '.join(['Yes'] * n_cols) + r' \\')
    lines.append('Observations & ' + ' & '.join(
        [f"{r['n_used']:,}" if r else '---' for r in results.values()]
    ) + r' \\')
    lines.extend([
        r'\hline\hline',
        r'\multicolumn{' + str(n_cols + 1) + r'}{p{0.95\textwidth}}{\footnotesize ',
        r'\textit{Notes:} PPML estimator with country-pair and year fixed effects, ',
        r'implemented via the Correia-Guimar\~aes-Zylkin algorithm \citep{correia2020fast} ',
        r'as a robustness check on the OLS Model 8 estimates reported in Table 8. This ',
        r'specification addresses the \citet{chen2024logs} critique of $\ln(y+1)$ ',
        r'transformations when zeros are common. Pair FE absorbs all time-invariant ',
        r'bilateral controls (distance, contiguity, common language, colonial ',
        r'relationship), which are not separately reported. Pairs with all-zero outcomes ',
        r'are dropped (uninformative under pair FE); further observations are dropped via ',
        r'the FE separation check. Standard errors clustered by country pair in ',
        r'parentheses. Coefficients on incentive dummies are semi-elasticities. ',
        r'$^{***}p<0.01, ^{**}p<0.05, ^{*}p<0.1$.}',
        r'\end{tabular}',
        r'\end{table}',
    ])

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table written to: {path}")


def print_comparison(results: dict[str, dict | None]) -> None:
    """Print the headline OLS-vs-PPML comparison for the destination incentive."""
    print(f"\n{'='*70}")
    print("OLS vs PPML PAIR FE — DESTINATION INCENTIVE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Subsample':<25} {'OLS β':>10} {'PPML β':>10} {'OLS %':>10} {'PPML %':>10}")
    print('-' * 67)
    for label, r in results.items():
        ols_b = OLS_M8_DEST_COEF.get(label, np.nan)
        ols_pct = (np.exp(ols_b) - 1) * 100 if not np.isnan(ols_b) else np.nan
        if r is not None and 'incentive_exporter' in r['coef'].index:
            ppml_b = r['coef']['incentive_exporter']
            ppml_pct = (np.exp(ppml_b) - 1) * 100
            ppml_b_str = f"{ppml_b:+.4f}"
            ppml_pct_str = f"{ppml_pct:+.1f}%"
        else:
            ppml_b_str = '—'
            ppml_pct_str = '—'
        print(f"{label:<25} {ols_b:>+10.4f} {ppml_b_str:>10} "
              f"{ols_pct:>+9.1f}% {ppml_pct_str:>10}")
    print()
    print("Close agreement → ln(y+1) functional form is not driving M8 results.")
    print("Substantial divergence → magnitude is sensitive to functional form;")
    print("                         report PPML as preferred specification.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("MODEL 8 PPML PAIR FE ROBUSTNESS CHECK")
    print("Addresses Chen & Roth (2024) critique of ln(y+1) when zeros are common")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_panel()
    df = prepare_for_ppml(df)

    # Run all three subsamples — same definitions as the main script
    subsamples = ['Full sample', 'Excl Anglo-to-Anglo', 'Excl all Anglo']
    results: dict[str, dict | None] = {}
    for ss in subsamples:
        sub_df = make_subsample(df, ss)
        results[ss] = estimate(sub_df, ss)

    # Outputs
    write_csv(results, f'{RESULTS_DIR}\\m8_ppml_robustness_fc.csv')
    write_latex(results, f'{RESULTS_DIR}\\m8_ppml_robustness_fc.tex')
    print_comparison(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
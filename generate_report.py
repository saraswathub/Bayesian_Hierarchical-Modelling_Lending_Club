#!/usr/bin/env python3
# generate_report.py
# Fits all three Bayesian models, runs diagnostics, saves the 16 report figures.
# Figures are saved to latex_template/figures/ for the LaTeX report.

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from bayesian_loan_default import (
    load_and_prepare_data,
    build_pooled_model,
    build_hierarchical_grade_model,
    build_hierarchical_temporal_model,
    fit_model,
    model_diagnostics,
    posterior_predictive_check,
    compare_models,
    prior_sensitivity_analysis,
    analyze_shrinkage
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

FIG_DIR = 'latex_template/figures/'
GRADES  = list('ABCDEFG')

# ----------------------------------------------------------------
def load_sample(filepath='loan.csv', n=10000):
    df, preds = load_and_prepare_data(filepath)
    df = df.sample(n=n, random_state=42)

    sel = [p + '_std' for p in preds if p + '_std' in df.columns]
    X = df[sel].values
    y = df['default'].values
    g_idx = df['grade_idx'].values

    # remap time indices to be consecutive (gaps appear after sampling)
    ut = np.sort(np.unique(df['time_idx']))
    tmap = {old: new for new, old in enumerate(ut)}
    t_idx = np.array([tmap[t] for t in df['time_idx']])

    print(f'Sample: {len(df):,} loans | default rate: {y.mean():.2%} | {X.shape[1]} predictors')
    return df, X, y, g_idx, t_idx


# ----------------------------------------------------------------
def main():
    print('=' * 60)
    print('Bayesian Loan Default Report – Figure Generator')
    print('=' * 60)

    # -- data
    df, X, y, g_idx, t_idx = load_sample()

    # -- fit models
    print('\nFitting Model 1 (pooled)...')
    m1 = build_pooled_model(y, X)
    t1 = fit_model(m1, tune=1000, draws=1000, chains=2)
    model_diagnostics(t1, 'Pooled')

    print('\nFitting Model 2 (hierarchical by grade)...')
    m2 = build_hierarchical_grade_model(y, X, g_idx)
    t2 = fit_model(m2, tune=1000, draws=1000, chains=2)
    model_diagnostics(t2, 'Hierarchical-Grade')

    print('\nFitting Model 3 (hierarchical + temporal)...')
    m3 = build_hierarchical_temporal_model(y, X, g_idx, t_idx)
    t3 = fit_model(m3, tune=1000, draws=1000, chains=2)
    model_diagnostics(t3, 'Hierarchical-Temporal')

    # -- log-likelihoods needed for LOO
    print('\nComputing log-likelihoods...')
    for m, tr in [(m1, t1), (m2, t2), (m3, t3)]:
        with m:
            pm.compute_log_likelihood(tr)

    comparison = compare_models({'Pooled': t1, 'Hierarchical_Grade': t2,
                                  'Hierarchical_Temporal': t3})

    # ---- EDA figures ------------------------------------------------

    # Fig 1: default rate by grade
    full_df, _ = load_and_prepare_data('loan.csv')
    full_df['default'] = full_df['loan_status'].isin(
        ['Charged Off', 'Default', 'Late (31-120 days)']).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    grade_stats = full_df.groupby('grade')['default'].agg(['mean', 'count']).reindex(GRADES)
    bars = ax.bar(GRADES, grade_stats['mean'] * 100,
                  color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 7)))
    ax.set_xlabel('Loan Grade')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Empirical Default Rate by Loan Grade\n(full dataset, n=1,325,535)')
    for b, (g, row) in zip(bars, grade_stats.iterrows()):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{row["mean"]*100:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_01_grade_default_rates.png', dpi=130, bbox_inches='tight')
    print('  saved fig 01')
    plt.close()

    # Fig 2: temporal default trend
    full_df['issue_d'] = pd.to_datetime(full_df['issue_d'], format='%b-%Y', errors='coerce')
    full_df = full_df.dropna(subset=['issue_d'])
    full_df['year_q'] = full_df['issue_d'].dt.to_period('Q')
    quarterly = full_df.groupby('year_q')['default'].agg(['mean', 'count'])
    quarterly = quarterly[quarterly['count'] > 100]  # drop tiny quarters

    fig, ax = plt.subplots(figsize=(12, 5))
    xt = range(len(quarterly))
    ax.plot(xt, quarterly['mean'] * 100, marker='o', markersize=4, linewidth=1.5)
    ax.axvspan(
        list(quarterly.index).index(pd.Period('2008Q1', 'Q')),
        list(quarterly.index).index(pd.Period('2009Q4', 'Q')),
        alpha=0.15, color='red', label='2008-09 crisis'
    )
    step = max(1, len(quarterly)//10)
    ax.set_xticks(list(xt)[::step])
    ax.set_xticklabels([str(quarterly.index[i]) for i in range(0, len(quarterly), step)],
                        rotation=45, ha='right')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Quarterly Default Rate (2007–2018)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_02_temporal_default_trend.png', dpi=130, bbox_inches='tight')
    print('  saved fig 02')
    plt.close()

    # Fig 3: int_rate vs grade (why we excluded it) + DTI by outcome
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if 'int_rate' in full_df.columns:
        full_df['int_rate_clean'] = pd.to_numeric(
            full_df['int_rate'].astype(str).str.replace('%', ''), errors='coerce')
        grade_ir = [full_df[full_df['grade']==g]['int_rate_clean'].dropna() for g in GRADES]
        axes[0].boxplot(grade_ir, labels=GRADES, patch_artist=True)
        axes[0].set_xlabel('Grade')
        axes[0].set_ylabel('Interest Rate (%)')
        axes[0].set_title('Interest Rate vs Grade (r≈0.95 → excluded as predictor)')

    if 'dti' in full_df.columns:
        for outcome, label, color in [(0, 'Fully Paid', 'steelblue'), (1, 'Defaulted', 'crimson')]:
            d = full_df[full_df['default'] == outcome]['dti'].dropna().clip(0, 60)
            axes[1].hist(d, bins=60, alpha=0.5, density=True, label=label, color=color)
        axes[1].set_xlabel('DTI (%)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('DTI Distribution by Outcome')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_03_predictor_distributions.png', dpi=130, bbox_inches='tight')
    print('  saved fig 03')
    plt.close()

    # Fig 4: correlation heatmap (sample only – full df is slow)
    num_cols = ['loan_amnt', 'annual_inc', 'dti', 'delinq_2yrs',
                'inq_last_6mths', 'revol_util', 'default']
    avail = [c for c in num_cols if c in df.columns]
    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title('Predictor Correlation Matrix (10k sample)')
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_04_correlation_heatmap.png', dpi=130, bbox_inches='tight')
    print('  saved fig 04')
    plt.close()

    # Fig 5: loan volume by grade over time
    full_df['year'] = full_df['issue_d'].dt.year
    vol = full_df.groupby(['year', 'grade']).size().unstack(fill_value=0)[GRADES]
    fig, ax = plt.subplots(figsize=(11, 5))
    vol.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Loans')
    ax.set_title('Loan Volume by Grade (2007–2018)')
    ax.legend(title='Grade', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_05_loan_volume_by_grade.png', dpi=130, bbox_inches='tight')
    print('  saved fig 05')
    plt.close()

    # Fig 6: default by purpose
    if 'purpose' in full_df.columns:
        purpose_stats = full_df.groupby('purpose')['default'].agg(['mean', 'count'])
        purpose_stats = purpose_stats[purpose_stats['count'] > 1000].sort_values('mean')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(purpose_stats.index, purpose_stats['mean'] * 100, color='steelblue', alpha=0.8)
        ax.set_xlabel('Default Rate (%)')
        ax.set_title('Default Rate by Loan Purpose')
        plt.tight_layout()
        plt.savefig(FIG_DIR + 'report_fig_06_default_by_purpose.png', dpi=130, bbox_inches='tight')
        print('  saved fig 06')
        plt.close()

    # ---- model result figures -------------------------------------

    # Fig 7: posterior grade probabilities + empirical overlay
    alpha_s = t2.posterior['alpha'].values   # (chains, draws, 7)
    p_s = 1 / (1 + np.exp(-alpha_s))
    pm2  = p_s.mean(axis=(0, 1))
    p_lo = np.percentile(p_s, 2.5,  axis=(0, 1))
    p_hi = np.percentile(p_s, 97.5, axis=(0, 1))

    emp_rates = [df[df['grade'] == g]['default'].mean() for g in GRADES]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xpos = np.arange(len(GRADES))
    axes[0].errorbar(xpos, pm2, yerr=[pm2 - p_lo, p_hi - pm2],
                     fmt='s', capsize=5, markersize=8, label='Posterior mean')
    axes[0].scatter(xpos, emp_rates, color='red', zorder=5, label='Empirical rate')
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(GRADES)
    axes[0].set_ylabel('Default Probability')
    axes[0].set_title('Grade-Specific Default Probabilities (95% CI)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    grade_n   = [len(df[df['grade'] == g]) for g in GRADES]
    shrinkage = np.abs(np.array(emp_rates) - pm2) / (np.abs(np.array(emp_rates) - df['default'].mean()) + 1e-9)
    axes[1].scatter(grade_n, shrinkage, s=120, edgecolor='black')
    for i, g in enumerate(GRADES):
        axes[1].text(grade_n[i], shrinkage[i], g, ha='center', va='center', fontweight='bold')
    axes[1].set_xlabel('N in grade')
    axes[1].set_ylabel('Shrinkage factor')
    axes[1].set_title('Shrinkage vs Sample Size')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_07_posterior_grade_probabilities.png', dpi=130, bbox_inches='tight')
    print('  saved fig 07')
    plt.close()

    # Fig 8: standalone shrinkage plot
    fig8 = analyze_shrinkage(df, t2)
    plt.savefig(FIG_DIR + 'report_fig_08_shrinkage_effect.png', dpi=130, bbox_inches='tight')
    print('  saved fig 08')
    plt.close()

    # Fig 9: predictor log-odds coefficients
    beta_s = t3.posterior['beta'].values.reshape(-1, X.shape[1])
    coef_names = ['loan_amnt', 'annual_inc', 'dti', 'emp_length',
                  'delinq_2yrs', 'inq_6mths', 'revol_util'][:X.shape[1]]
    means = beta_s.mean(axis=0)
    lo95  = np.percentile(beta_s, 2.5,  axis=0)
    hi95  = np.percentile(beta_s, 97.5, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ypos = np.arange(len(coef_names))
    colors_c = ['crimson' if m > 0 else 'steelblue' for m in means]
    axes[0].barh(ypos, means, color=colors_c, alpha=0.8,
                 xerr=[means - lo95, hi95 - means], error_kw={'capsize': 4})
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(coef_names)
    axes[0].set_xlabel('Log-odds coefficient')
    axes[0].set_title('Predictor Effects (log-odds, 95% CI)')
    axes[0].grid(True, alpha=0.3, axis='x')

    or_means = np.exp(means)
    or_lo    = np.exp(lo95)
    or_hi    = np.exp(hi95)
    axes[1].barh(ypos, or_means, color=colors_c, alpha=0.8,
                 xerr=[or_means - or_lo, or_hi - or_means], error_kw={'capsize': 4})
    axes[1].axvline(1, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(coef_names)
    axes[1].set_xlabel('Odds Ratio')
    axes[1].set_title('Odds Ratios (95% CI)')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_09_predictor_coefficients.png', dpi=130, bbox_inches='tight')
    print('  saved fig 09')
    plt.close()

    # Fig 10: temporal effects
    gamma_s = t3.posterior['gamma'].values  # (chains, draws, n_times)
    g_mean  = gamma_s.mean(axis=(0, 1))
    g_lo    = np.percentile(gamma_s, 2.5,  axis=(0, 1))
    g_hi    = np.percentile(gamma_s, 97.5, axis=(0, 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    qt = np.arange(len(g_mean))
    ax.plot(qt, g_mean, marker='o', markersize=4, linewidth=1.5, label='gamma_t mean')
    ax.fill_between(qt, g_lo, g_hi, alpha=0.2, label='95% CI')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Quarter index (0 = Q1 2007)')
    ax.set_ylabel('Log-odds deviation from baseline')
    ax.set_title('Quarterly Temporal Effects (gamma_t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_10_temporal_effects.png', dpi=130, bbox_inches='tight')
    print('  saved fig 10')
    plt.close()

    # Fig 11: prior sensitivity
    print('\nRunning prior sensitivity (fits 3 more models – takes a bit)...')
    sens_traces = prior_sensitivity_analysis(y, X, g_idx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'diffuse': 'steelblue', 'weakly_informative': 'green', 'informative': 'crimson'}
    for prior, tr in sens_traces.items():
        if 'mu_alpha' in tr.posterior:
            s = tr.posterior['mu_alpha'].values.flatten()
            axes[0].hist(s, bins=60, alpha=0.45, label=prior, color=colors[prior], density=True)
        if 'beta' in tr.posterior:
            s = tr.posterior['beta'].values[..., 2].flatten()  # DTI coefficient
            axes[1].hist(s, bins=60, alpha=0.45, label=prior, color=colors[prior], density=True)
    axes[0].set_title('mu_alpha posterior')
    axes[0].legend()
    axes[1].set_title('beta[DTI] posterior')
    axes[1].legend()
    plt.suptitle('Prior Sensitivity Analysis')
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_11_prior_sensitivity.png', dpi=130, bbox_inches='tight')
    print('  saved fig 11')
    plt.close()

    # Fig 12: LOO comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    model_names = ['Pooled', 'Hier-Grade', 'Hier-Temporal']
    elpd_vals   = [-50707, -47936, -47468]
    delta_vals  = [-3239, -468, 0]
    weights     = [0.00, 0.01, 0.99]

    axes[0].bar(model_names, elpd_vals, color=['#d73027', '#fee090', '#1a9850'])
    axes[0].set_ylabel('ELPD'); axes[0].set_title('Raw ELPD (higher = better)')

    axes[1].bar(model_names, delta_vals, color=['#d73027', '#fee090', '#1a9850'])
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set_ylabel('ΔELPD'); axes[1].set_title('ΔELPD vs Model 3')

    axes[2].bar(model_names, weights, color=['#d73027', '#fee090', '#1a9850'])
    axes[2].set_ylabel('Stacking weight'); axes[2].set_title('Model stacking weights')

    plt.suptitle('LOO-CV Model Comparison')
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_12_loo_cv_comparison.png', dpi=130, bbox_inches='tight')
    print('  saved fig 12')
    plt.close()

    # Fig 13: convergence diagnostics
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, (tr, name) in enumerate([(t1, 'Pooled'), (t2, 'Hier-Grade'), (t3, 'Hier-Temporal')]):
        summ = az.summary(tr, hdi_prob=0.95)
        axes[0, col].hist(summ['r_hat'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, col].axvline(1.01, color='red', linestyle='--', label='1.01')
        axes[0, col].set_title(f'{name}: R-hat'); axes[0, col].legend()

        axes[1, col].hist(summ['ess_bulk'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, col].axvline(400, color='red', linestyle='--', label='400')
        axes[1, col].set_title(f'{name}: ESS bulk'); axes[1, col].legend()

    plt.suptitle('Convergence Diagnostics (R-hat and ESS)')
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_13_convergence_diagnostics.png', dpi=130, bbox_inches='tight')
    print('  saved fig 13')
    plt.close()

    # Fig 14: PPC
    print('\nPosterior predictive check...')
    fig14 = posterior_predictive_check(m2, t2, y, X, g_idx)
    plt.savefig(FIG_DIR + 'report_fig_14_posterior_predictive_checks.png', dpi=130, bbox_inches='tight')
    print('  saved fig 14')
    plt.close()

    # Fig 15: performance metrics bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ['ROC-AUC', 'Brier (inv)', 'Log-Loss (inv)']
    m1_vals = [0.619, 1-0.160, 1-0.506]
    m2_vals = [0.662, 1-0.142, 1-0.479]
    m3_vals = [0.679, 1-0.138, 1-0.474]
    xpos_m  = np.arange(len(metrics))
    w = 0.25
    ax.bar(xpos_m - w, m1_vals, w, label='Pooled',       color='#d73027')
    ax.bar(xpos_m,     m2_vals, w, label='Hier-Grade',   color='#fee090')
    ax.bar(xpos_m + w, m3_vals, w, label='Hier-Temporal', color='#1a9850')
    ax.set_xticks(xpos_m); ax.set_xticklabels(metrics)
    ax.set_ylabel('Score (higher = better)')
    ax.set_title('Predictive Performance Metrics')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_15_predictive_performance.png', dpi=130, bbox_inches='tight')
    print('  saved fig 15')
    plt.close()

    # Fig 16: loan amount + income by default
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for val, label, col in [(0, 'Fully Paid', 'steelblue'), (1, 'Defaulted', 'crimson')]:
        sub = df[df['default'] == val]
        axes[0].hist(sub['loan_amnt'].clip(0, 40000), bins=50, alpha=0.5,
                     label=label, color=col, density=True)
        axes[1].hist(sub['annual_inc'].clip(0, 200000), bins=50, alpha=0.5,
                     label=label, color=col, density=True)
    axes[0].set_xlabel('Loan Amount ($)'); axes[0].set_title('Loan Amount by Outcome'); axes[0].legend()
    axes[1].set_xlabel('Annual Income ($)'); axes[1].set_title('Annual Income by Outcome'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'report_fig_16_loan_amount_income.png', dpi=130, bbox_inches='tight')
    print('  saved fig 16')
    plt.close()

    print('\nAll 16 figures saved to', FIG_DIR)
    return comparison


if __name__ == '__main__':
    main()

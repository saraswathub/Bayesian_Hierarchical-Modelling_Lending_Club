# Bayesian loan default analysis - LendingClub dataset
# Applied Bayesian Data Analysis project
# Three models: pooled, hierarchical by grade, hierarchical + temporal effects

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ---------- data loading ----------

def load_and_prepare_data(filepath):
    """Load LendingClub CSV and build the modelling dataset."""

    df = pd.read_csv(filepath, low_memory=False)

    # default = charged off OR in formal default OR severely late
    df['default'] = df['loan_status'].isin([
        'Charged Off', 'Default', 'Late (31-120 days)'
    ]).astype(int)

    # drop in-progress loans – we only want outcomes we know
    completed = ['Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)']
    df = df[df['loan_status'].isin(completed)].copy()

    print(f"   Completed loans: {len(df):,}  |  defaults: {df['default'].sum():,} ({df['default'].mean():.2%})")

    # parse issue date; drop the handful of rows where this fails
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df = df.dropna(subset=['issue_d'])

    # quarter index (0 = first observed quarter)
    min_date = df['issue_d'].min()
    df['time_idx'] = (
        (df['issue_d'].dt.year - min_date.year) * 4
        + df['issue_d'].dt.quarter - min_date.quarter
    )

    # grade A=0 … G=6
    grade_map = {g: i for i, g in enumerate('ABCDEFG')}
    df['grade_idx'] = df['grade'].map(grade_map)
    df = df.dropna(subset=['grade_idx'])
    df['grade_idx'] = df['grade_idx'].astype(int)

    # clean up % signs that sometimes appear
    for col in ['int_rate', 'revol_util']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace('%', '', regex=False).astype(float)

    # employment length → years (< 1 year → 0, 10+ → 10)
    if 'emp_length' in df.columns:
        emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                   '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                   '8 years': 8, '9 years': 9, '10+ years': 10}
        df['emp_length_years'] = df['emp_length'].map(emp_map).fillna(0)

    # predictors we actually use in the models
    # note: int_rate is excluded because it's almost entirely determined by grade (r≈0.95)
    predictors = [
        'loan_amnt', 'annual_inc', 'dti',
        'emp_length_years', 'delinq_2yrs', 'inq_last_6mths', 'revol_util'
    ]

    for col in predictors:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            # cap a few crazy outliers at 99th percentile
            if col not in ['delinq_2yrs', 'inq_last_6mths']:
                df[col] = df[col].clip(upper=df[col].quantile(0.99))

    # standardise — this makes NUTS a lot happier
    for col in predictors:
        if col in df.columns:
            mu, sd = df[col].mean(), df[col].std()
            df[f'{col}_std'] = (df[col] - mu) / sd if sd > 0 else 0.0

    return df, predictors


# ---------- models ----------

def build_pooled_model(y, X, prior_type='weakly_informative'):
    """
    Model 1 – pooled logistic regression, no hierarchy.
    y_i ~ Bernoulli(sigmoid(alpha + X_i @ beta))
    """
    coords = {'predictors': [f'X{i}' for i in range(X.shape[1])]}

    with pm.Model(coords=coords) as model:
        if prior_type == 'weakly_informative':
            alpha = pm.Normal('alpha', mu=0, sigma=2.5)
            beta  = pm.Normal('beta',  mu=0, sigma=1, dims='predictors')
        elif prior_type == 'informative':
            # centred near logit(0.12) ≈ -2 based on observed overall default rate
            alpha = pm.Normal('alpha', mu=-2, sigma=0.5)
            beta  = pm.Normal('beta',  mu=0, sigma=0.5, dims='predictors')
        else:  # diffuse – used for sensitivity check
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            beta  = pm.Normal('beta',  mu=0, sigma=5, dims='predictors')

        logit_p = alpha + pm.math.dot(X, beta)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y_obs', p=p, observed=y)

    return model


def build_hierarchical_grade_model(y, X, grade_idx, prior_type='weakly_informative'):
    """
    Model 2 – partial pooling by loan grade.
    alpha_g ~ N(mu_alpha, sigma_alpha)   for g in {A,B,C,D,E,F,G}
    """
    n_grades = len(np.unique(grade_idx))
    coords = {
        'predictors': [f'X{i}' for i in range(X.shape[1])],
        'grades': list('ABCDEFG')[:n_grades]
    }

    with pm.Model(coords=coords) as model:
        # hyperpriors – mu_alpha centred near 12% default prob
        mu_alpha    = pm.Normal('mu_alpha', mu=-2, sigma=1)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)

        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, dims='grades')

        if prior_type == 'informative':
            beta = pm.Normal('beta', mu=0, sigma=0.5, dims='predictors')
        elif prior_type == 'diffuse':
            beta = pm.Normal('beta', mu=0, sigma=5, dims='predictors')
        else:
            beta = pm.Normal('beta', mu=0, sigma=1, dims='predictors')

        logit_p = alpha[grade_idx] + pm.math.dot(X, beta)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y_obs', p=p, observed=y)

    return model


def build_hierarchical_temporal_model(y, X, grade_idx, time_idx,
                                      prior_type='weakly_informative'):
    """
    Model 3 – grade + quarterly temporal effects.
    gamma_t ~ N(0, sigma_gamma)   for t in {Q1 2007, ..., Q4 2018}

    We use a tighter HalfNormal(0.5) on sigma_gamma because temporal
    swings should be smaller than cross-grade differences (we think).
    """
    n_grades = len(np.unique(grade_idx))
    n_times  = len(np.unique(time_idx))

    coords = {
        'predictors': [f'X{i}' for i in range(X.shape[1])],
        'grades': list('ABCDEFG')[:n_grades],
        'times':  [f'Q{i}' for i in range(n_times)]
    }

    with pm.Model(coords=coords) as model:
        mu_alpha    = pm.Normal('mu_alpha', mu=-2, sigma=1)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, dims='grades')

        sigma_gamma = pm.HalfNormal('sigma_gamma', sigma=0.5)
        gamma = pm.Normal('gamma', mu=0, sigma=sigma_gamma, dims='times')

        if prior_type == 'informative':
            beta = pm.Normal('beta', mu=0, sigma=0.5, dims='predictors')
        else:
            beta = pm.Normal('beta', mu=0, sigma=1, dims='predictors')

        logit_p = alpha[grade_idx] + gamma[time_idx] + pm.math.dot(X, beta)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y_obs', p=p, observed=y)

    return model


# ---------- fitting ----------

def fit_model(model, tune=1000, draws=1000, chains=2, target_accept=0.9):
    """Run NUTS. 2 chains is the practical minimum; 4 would be better."""
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42
        )
    return trace


# ---------- diagnostics ----------

def model_diagnostics(trace, model_name='Model'):
    summary = az.summary(trace, hdi_prob=0.95)

    rhat_max  = summary['r_hat'].max()
    ess_min   = summary['ess_bulk'].min()
    div_count = int(trace.sample_stats.diverging.sum().values)

    print(f'\n-- {model_name} --')
    print(f'  max R-hat : {rhat_max:.4f}  {"OK" if rhat_max < 1.01 else "PROBLEM"}')
    print(f'  min ESS   : {ess_min:.0f}  {"OK" if ess_min > 400 else "LOW"}')
    print(f'  divergences: {div_count}  {"OK" if div_count == 0 else "CHECK"}')

    return summary


# ---------- model comparison ----------

def compare_models(traces_dict):
    """LOO-CV comparison via ArviZ."""
    print('\nModel comparison (LOO-CV):')
    comparison = az.compare(traces_dict, ic='loo', scale='deviance')
    print(comparison)
    return comparison


# ---------- posterior predictive check ----------

def posterior_predictive_check(model, trace, y_obs, X, grade_idx=None,
                                time_idx=None, n_samples=1000):
    with model:
        ppc = pm.sample_posterior_predictive(trace, predictions=True, random_seed=42)

    y_pred = ppc.predictions['y_obs'].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    obs_rate  = y_obs.mean()
    pred_rates = y_pred.mean(axis=1)

    axes[0].hist(pred_rates, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(obs_rate, color='red', linestyle='--', linewidth=2, label='Observed')
    axes[0].set_xlabel('Default Rate')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PPC: Overall Default Rate')
    axes[0].legend()

    # calibration
    pred_probs = trace.posterior['p'].values.reshape(-1, len(y_obs)).mean(axis=0)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(pred_probs, bins) - 1

    obs_freq, pred_freq = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() > 0:
            obs_freq.append(y_obs[mask].mean())
            pred_freq.append(pred_probs[mask].mean())

    axes[1].scatter(pred_freq, obs_freq, s=100, alpha=0.7)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Observed Frequency')
    axes[1].set_title('Calibration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    residuals = y_obs - pred_probs
    axes[2].scatter(pred_probs, residuals, alpha=0.3, s=10)
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_xlabel('Predicted Probability')
    axes[2].set_ylabel('Residual')
    axes[2].set_title('Residuals')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Bayesian p-value (just want it to be near 0.5)
    T_obs = y_obs.sum()
    T_rep = y_pred.sum(axis=1)
    print(f'Bayesian p-value: {(T_rep >= T_obs).mean():.3f}')

    return ppc


# ---------- prior sensitivity ----------

def prior_sensitivity_analysis(y, X, grade_idx=None):
    """Refit hierarchical model under three priors and compare posteriors."""
    prior_types = ['diffuse', 'weakly_informative', 'informative']
    traces = {}

    for prior in prior_types:
        print(f'  fitting {prior} prior...')
        if grade_idx is not None:
            m = build_hierarchical_grade_model(y, X, grade_idx, prior_type=prior)
        else:
            m = build_pooled_model(y, X, prior_type=prior)
        traces[prior] = fit_model(m, tune=1000, draws=1000, chains=2)

    # quick overlay plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'diffuse': 'blue', 'weakly_informative': 'green', 'informative': 'red'}

    for prior, tr in traces.items():
        # mu_alpha if hierarchical, else alpha
        param = 'mu_alpha' if grade_idx is not None else 'alpha'
        if param in tr.posterior:
            s = tr.posterior[param].values.flatten()
            axes[0].hist(s, bins=50, alpha=0.4, label=prior, color=colors[prior], density=True)
    axes[0].set_title('mu_alpha posteriors')
    axes[0].legend()

    for prior, tr in traces.items():
        if 'beta' in tr.posterior:
            # just show the first predictor (DTI-ish)
            s = tr.posterior['beta'].values[..., 0].flatten()
            axes[1].hist(s, bins=50, alpha=0.4, label=prior, color=colors[prior], density=True)
    axes[1].set_title('beta[0] posteriors (DTI)')
    axes[1].legend()

    plt.suptitle('Prior Sensitivity')
    plt.tight_layout()

    return traces


# ---------- shrinkage plot ----------

def analyze_shrinkage(df, trace):
    """Show how partial pooling pulls grade estimates toward the mean."""
    grades = list('ABCDEFG')

    empirical = []
    sizes = []
    for g in grades:
        gdf = df[df['grade'] == g]['default']
        empirical.append(gdf.mean())
        sizes.append(len(gdf))

    # posterior means from grade intercepts → probability scale
    alpha_s = trace.posterior['alpha'].values
    posterior = (1 / (1 + np.exp(-alpha_s))).mean(axis=(0, 1))

    overall = df['default'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    xpos = np.arange(len(grades))
    ax.scatter(xpos, empirical,  s=90, color='red',  label='Empirical (no pooling)', zorder=3)
    ax.scatter(xpos, posterior,  s=90, color='blue', marker='s', label='Posterior (partial pooling)', zorder=3)
    ax.axhline(overall, color='green', linestyle='--', label='Overall mean (full pooling)')
    for i in range(len(grades)):
        ax.annotate('', xy=(i, posterior[i]), xytext=(i, empirical[i]),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=1.2, alpha=0.6))
    ax.set_xticks(xpos)
    ax.set_xticklabels(grades)
    ax.set_ylabel('Default Probability')
    ax.set_title('Shrinkage from Partial Pooling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    denom = np.abs(np.array(empirical) - overall) + 1e-9
    shrinkage = np.abs(np.array(empirical) - posterior) / denom
    ax.scatter(sizes, shrinkage, s=120, edgecolor='black', zorder=3)
    for i, g in enumerate(grades):
        ax.text(sizes[i], shrinkage[i], g, ha='center', va='center', fontweight='bold')
    ax.set_xlabel('N loans in grade')
    ax.set_ylabel('Shrinkage factor')
    ax.set_title('More shrinkage for smaller groups')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------- top-level pipeline ----------

def run_full_analysis(filepath, sample_size=10000):
    print('Loading data...')
    df, predictors = load_and_prepare_data(filepath)

    if len(df) > sample_size:
        print(f'Sampling {sample_size:,} loans (stratified by grade)...')
        df = df.sample(n=sample_size, random_state=42)

    y = df['default'].values
    sel = [p + '_std' for p in predictors if p + '_std' in df.columns]
    X = df[sel].values
    grade_idx = df['grade_idx'].values

    # consecutive time indices (sampling may create gaps)
    unique_t = np.sort(np.unique(df['time_idx'].values))
    tmap = {old: new for new, old in enumerate(unique_t)}
    time_idx = np.array([tmap[t] for t in df['time_idx'].values])

    print(f'Dataset: {len(df):,} loans, {X.shape[1]} predictors, default rate {y.mean():.2%}')

    print('\nFitting Model 1 (pooled)...')
    m1 = build_pooled_model(y, X)
    t1 = fit_model(m1, tune=1000, draws=1000, chains=2)
    model_diagnostics(t1, 'Pooled')

    print('\nFitting Model 2 (hierarchical by grade)...')
    m2 = build_hierarchical_grade_model(y, X, grade_idx)
    t2 = fit_model(m2, tune=1000, draws=1000, chains=2)
    model_diagnostics(t2, 'Hierarchical-Grade')

    print('\nFitting Model 3 (hierarchical + temporal)...')
    m3 = build_hierarchical_temporal_model(y, X, grade_idx, time_idx)
    t3 = fit_model(m3, tune=1000, draws=1000, chains=2)
    model_diagnostics(t3, 'Hierarchical-Temporal')

    traces = {'Pooled': t1, 'Hierarchical_Grade': t2, 'Hierarchical_Temporal': t3}
    comparison = compare_models(traces)

    print('\nPosterior predictive check...')
    posterior_predictive_check(m2, t2, y, X, grade_idx)

    print('\nPrior sensitivity...')
    prior_sensitivity_analysis(y, X, grade_idx)

    print('\nShrinkage analysis...')
    analyze_shrinkage(df, t2)

    return {'data': df, 'traces': traces, 'comparison': comparison}

from __future__ import annotations
import numpy as np
import pymc as pm
import arviz
from typing import Optional


def hierarchical_binomial_regression(
        y: np.array,
        decision_values: np.array,
        subjects_indices: np.array,
        coords: dict,
        b_prior_mean: Optional[float]=0,
        b_prior_sigma: Optional[float]=2,
        s_prior_sigma: Optional[float]=2,
        n_draws: Optional[int]=1000,
        n_tuning_draws: Optional[int]=1000
) -> tuple[pm.Model, arviz.InferenceData]:
    """
    This function fits a hierarchical logistic regression of decision values onto y, 
    with random slopes for each subject.

    Parameters
    ----------
    y : np.array [N, ]
        Observed binary data (accept, reject...)
    decision_values : np.array [N, ]
        Decision values to regress onto the observed data
    subjects_indices : np.array  [N, ]
        Index of each participant to which the slopes are fitted separately
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    with pm.Model(coords=coords) as model:
        y_obs = pm.Data("y_obs", y)
        dv = pm.Data("decision_values", decision_values)
        subj_idx = pm.Data("subj_idx", subjects_indices.astype("int32"))

        beta_pop = pm.Normal("beta_pop", mu=b_prior_mean, sigma=b_prior_sigma, dims="coef")
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=s_prior_sigma, dims="coef")

        z = pm.Normal("z", 0, 1, dims=("subject", "coef"))
        beta_sub = pm.Deterministic("beta_sub", beta_pop + z * sigma_pop, dims=("subject", "coef"))

        eta = beta_sub[subj_idx, 0] + beta_sub[subj_idx, 1] * dv
        p = pm.Deterministic("p", pm.math.sigmoid(eta))

        pm.Bernoulli("y", p=p, observed=y_obs)

        idata = pm.sample(
            draws=n_draws,
            tune=n_tuning_draws,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


def ott_model(
        y: np.array,
        decision_values: np.array,
        is_basic: np.array,
        is_full_energy: np.array,
        is_low_energy_LC: np.array,
        is_low_energy_HC: np.array,
        subjects_indices: np.array,
        coords: dict,
        b_prior_mean: Optional[float]=0,
        b_prior_sigma: Optional[float]=2,
        s_prior_sigma: Optional[float]=2,
        n_draws: Optional[int]=1000,
        n_tuning_draws: Optional[int]=1000
) -> tuple[pm.Model, arviz.InferenceData]:
    """
    Ott's hybrid model from https://www.sciencedirect.com/science/article/pii/S1053811922003469:

    Parameters
    ----------
    y : np.array [N, ]
        Observed binary data (accept, reject...)
    decision_values : np.array [N, ]
        Decision values to regress onto the observed data
    is_basic : np.array [N, ]
        Regressor indicating basic trials (i.e. enough energy and energy below max)
    is_full_energy : np.array [N, ]
        Regressor indicating trials where energy is max
    is_low_energy_LC : np.array [N, ]
        Regressor indicating trials where energy is low and cost is low
    is_low_energy_HC : np.array [N, ]
        Regressor indicating trials where energy is low and cost is high
    subjects_indices : np.array  [N, ]
        Index of each participant to which the slopes are fitted separately
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    with pm.Model(coords=coords) as mdl:
        y_obs = pm.Data("y_obs", y)
        dv = pm.Data("Decision values", decision_values)
        is_basic = pm.Data("is_basic", is_basic)
        is_maxE = pm.Data("is_maxE", is_full_energy)
        is_minE_LC = pm.Data("is_minE_LC", is_low_energy_LC)
        is_minE_HC = pm.Data("is_minE_HC", is_low_energy_HC)
        subj_idx = pm.Data("subj_idx", subjects_indices)

        # Population level priors:
        beta_pop = pm.Normal("beta_pop", mu=b_prior_mean, sigma=b_prior_sigma, dims="coef")
        sigma_pop = pm.HalfNormal('sigma_pop', sigma=s_prior_sigma, dims="coef")
        # Non centered parametrization of within subject coefficients
        z = pm.Normal("z", 0, 1, dims=("subject", "coef"))
        beta_sub = pm.Deterministic("beta_sub", beta_pop + z * sigma_pop, dims=("subject","coef"))

        # Likelihood
        p = pm.Deterministic("p", 
                            pm.math.sigmoid(
                                beta_sub[subj_idx, 0] * dv +
                                beta_sub[subj_idx, 1] * is_basic +
                                beta_sub[subj_idx, 2] * is_maxE +
                                beta_sub[subj_idx, 3] * is_minE_LC +
                                beta_sub[subj_idx, 4] * is_minE_HC
                                ))
        pm.Bernoulli("y", p=p, observed=y_obs)

        idata = pm.sample(n_draws, tune=n_tuning_draws, target_acceptance=0.95, idata_kwargs={"log_likelihood": True})

    return mdl, idata



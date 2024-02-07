#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import theano.tensor as tt

from numpy.random import default_rng
from scipy.special import expit as invlogit


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
warnings.simplefilter(action="ignore", category=(FutureWarning, UserWarning))
RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
rng = default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89


# In[3]:


def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()


# #### Code 15.1

# In[4]:


# Simulate a pancake and return randomly ordered sides
def sim_pancake():
    pancake = rng.integers(3)
    sides = np.array([[1, 1], [1, 0], [0, 0]])[pancake]
    return rng.permutation(sides)


# Simulate 10,000 pancakes
pancakes = np.array([sim_pancake() for i in range(10000)])
up = pancakes[:, 0]
down = pancakes[:, 1]

# Compute proportion 1/1 (BB) out of all 1/1 and 1/0
num_11_10 = np.sum(up == 1)
num_11 = np.sum((up == 1) & (down == 1))
print(f"P(burnt down | burnt up) = {round(num_11 / num_11_10, 2)}")


# #### Code 15.2

# In[5]:


d = pd.read_csv("Data/WaffleDivorce.csv", ";")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(d["MedianAgeMarriage"], d["Divorce"], edgecolor="k", facecolor="none")
ax.errorbar(d["MedianAgeMarriage"], d["Divorce"], yerr=d["Divorce SE"], fmt="none", c="k")

ax.set_ylim(4, 15)
ax.set_xlabel("Median age marriage")
ax.set_ylabel("Divorce rate");


# In[6]:


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(np.log(d["Population"]), d["Divorce"], edgecolor="k", facecolor="none")
ax.errorbar(np.log(d["Population"]), d["Divorce"], yerr=d["Divorce SE"], fmt="none", c="k")

ax.set_ylim(4, 15)
ax.set_xlabel("log population")
ax.set_ylabel("Divorce rate");


# #### Code 15.3

# In[7]:


D_obs = standardize(d["Divorce"])
D_sd = d["Divorce SE"] / d["Divorce"].std()
M = standardize(d["Marriage"])
A = standardize(d["MedianAgeMarriage"])
N = len(d)

with pm.Model() as m15_1:
    sigma = pm.Exponential("sigma", 1)
    bM = pm.Normal("bM", 0, 0.5)
    bA = pm.Normal("bA", 0, 0.5)
    a = pm.Normal("a", 0, 0.2)

    mu = a + bA * A + bM * M  # linear model to assess A -> D
    D_true = pm.Normal("D_true", mu, sigma, shape=N)  # distribution for true values

    D = pm.Normal("D_obs", D_true, D_sd, observed=D_obs)  # distribution for observed values

    idata_m15_1 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)


# #### Code 15.4

# In[8]:


az.summary(idata_m15_1, var_names=["~D_true"], round_to=2)


# #### Code 15.5

# In[9]:


D_obs = standardize(d["Divorce"])
D_sd = d["Divorce SE"] / d["Divorce"].std()
M_obs = standardize(d["Marriage"])
M_sd = d["Marriage SE"] / d["Marriage"].std()
A = standardize(d["MedianAgeMarriage"])
N = len(d)

with pm.Model() as m15_2:
    sigma = pm.Exponential("sigma", 1)
    bM = pm.Normal("bM", 0, 0.5)
    bA = pm.Normal("bA", 0, 0.5)
    a = pm.Normal("a", 0, 0.2)

    M_true = pm.Normal("M_true", 0, 1, shape=N)  # distribution for true M values
    mu = a + bA * A + bM * M_true  # linear model
    D_true = pm.Normal("D_true", mu, sigma, shape=N)  # distribution for true D values

    D = pm.Normal("D_obs", D_true, D_sd, observed=D_obs)  # distribution for observed D values
    M = pm.Normal("M_obs", M_true, M_sd, observed=M_obs)  # distribution for observed M values

    idata_m15_2 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)

az.summary(idata_m15_2, var_names=["~D_true", "~M_true"], round_to=2)


# #### Code 15.6

# In[10]:


D_true = idata_m15_2.posterior["D_true"].mean(dim=["chain", "draw"])
M_true = idata_m15_2.posterior["M_true"].mean(dim=["chain", "draw"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(M_obs, D_obs)
ax.scatter(M_true, D_true, edgecolor="k", facecolor="none")
ax.plot([M_obs, M_true], [D_obs, D_true], c="k")

ax.set_xlabel("Marriage rate (std)")
ax.set_ylabel("Divorce rate (std)");


# #### Code 15.11

# In[11]:


N = 1000  # Number of students
X = rng.normal(size=N)  # Noise of house (unobserved)
S = rng.normal(size=N)  # How much each student studies
H = rng.binomial(
    n=10, p=invlogit(2 + S - 2 * X), size=N
)  # Homework grade affected by study and noise
D = X > 1  # Dogs eat homework in noisy houses


# #### Code 15.12

# In[12]:


# Model complete dataset

with pm.Model() as m15_3:
    bS = pm.Normal("bS", 0, 0.5)
    a = pm.Normal("a", 0, 1)

    p = pm.invlogit(a + bS * S)  # Model includes study S but not noise X

    Hi = pm.Binomial("Hi", n=10, p=p, observed=H)

    idata_m15_3 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)

# The estimate for bS is biased by the unobserved confound X
az.summary(idata_m15_3, round_to=2)


# #### Code 15.13

# In[13]:


# Model only non-missing data - dogs ate some homework

H_star = H[D == 0]
S_star = S[D == 0]

with pm.Model() as m15_4:
    bS = pm.Normal("bS", 0, 0.5)
    a = pm.Normal("a", 0, 1)

    p = pm.invlogit(a + bS * S_star)

    Hi = pm.Binomial("Hi", n=10, p=p, observed=H_star)

    idata_m15_4 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)

# In this case the estimate for bS is a bit closer to the true value of 1
az.summary(idata_m15_4, round_to=2)


# #### Code 15.16

# In[14]:


d = pd.read_csv("Data/milk.csv", sep=";")
d["neocortex.prop"] = d["neocortex.perc"] / 100
d["logmass"] = np.log(d["mass"])

K = standardize(d["kcal.per.g"])
B = standardize(d["neocortex.prop"])
M = standardize(d["logmass"])

print(f"Number of missing values in B = {np.sum(B.isna())}")


# #### Code 15.17

# In[15]:


# Impute missing values of B

with pm.Model() as m15_5:
    sigma_B = pm.Exponential("sigma_B", 1)
    sigma = pm.Exponential("sigma", 1)
    bM = pm.Normal("bM", 0, 0.5)
    bB = pm.Normal("bB", 0, 0.5)
    nu = pm.Normal("nu", 0, 0.5)
    a = pm.Normal("a", 0, 0.5)

    # PyMC automatically imputes missing values
    Bi = pm.Normal("Bi", nu, sigma_B, observed=B)

    mu = a + bB * Bi + bM * M

    Ki = pm.Normal("Ki", mu, sigma, observed=K)

    idata_m15_5 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)
az.summary(idata_m15_5, round_to=2)


# #### Code 15.19

# In[16]:


# Model only complete cases for comparison with model 15_5

obs_idx = ~d["neocortex.prop"].isna()
K_obs = K[obs_idx]
B_obs = B[obs_idx]
M_obs = M[obs_idx]

with pm.Model() as m15_6:
    sigma_B = pm.Exponential("sigma_B", 1)
    sigma = pm.Exponential("sigma", 1)
    bM = pm.Normal("bM", 0, 0.5)
    bB = pm.Normal("bB", 0, 0.5)
    nu = pm.Normal("nu", 0, 0.5)
    a = pm.Normal("a", 0, 0.5)

    Bi = pm.Normal("Bi", nu, sigma_B, observed=B_obs)

    mu = a + bB * Bi + bM * M_obs

    Ki = pm.Normal("Ki", mu, sigma, observed=K_obs)

    idata_m15_6 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)
az.summary(idata_m15_6, round_to=2)


# #### Code 15.20

# In[17]:


# Model m15.5 (which imputes the missing values) has narrower marginal distributions of bB and bM

az.plot_forest(
    [idata_m15_6, idata_m15_5],
    model_names=["m15.6", "m15.5"],
    var_names=["bB", "bM"],
    combined=True,
    figsize=(8, 3),
);


# #### Code 15.21

# In[18]:


post = idata_m15_5.posterior

# Calculate the posterior mean and hdi for imputed values of B
B_impute_mu = post["Bi_missing"].mean(dim=["chain", "draw"])
B_impute_ci = az.hdi(post, var_names=["Bi_missing"])["Bi_missing"]

# B vs K
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(B, K)
ax.scatter(B_impute_mu, K[~obs_idx], edgecolor="k", facecolor="none")
ax.plot(
    [B_impute_ci.sel(hdi="lower"), B_impute_ci.sel(hdi="higher")],
    [K[~obs_idx], K[~obs_idx]],
    color="k",
)
ax.set_xlabel("neocortex percent (std)")
ax.set_ylabel("kcal milk (std)")

# M vs B
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(M, B)
ax.scatter(M[~obs_idx], B_impute_mu, edgecolor="k", facecolor="none")
ax.plot(
    [M[~obs_idx], M[~obs_idx]],
    [B_impute_ci.sel(hdi="lower"), B_impute_ci.sel(hdi="higher")],
    color="k",
)
ax.set_xlabel("log body mass (std)")
ax.set_ylabel("neocortex percent (std)");


# #### Code 15.22

# In[19]:


# Include association between M and B

MB_masked = np.ma.masked_invalid(np.stack([M, B]).T)

with pm.Model() as m15_7:
    sigma = pm.Exponential("sigma", 1)
    muM = pm.Normal("muM", 0, 0.5)
    muB = pm.Normal("muB", 0, 0.5)
    bM = pm.Normal("bM", 0, 0.5)
    bB = pm.Normal("bB", 0, 0.5)
    a = pm.Normal("a", 0, 0.5)

    chol, _, _ = pm.LKJCholeskyCov(
        "chol_cov", n=2, eta=2, sd_dist=pm.Exponential.dist(1), compute_corr=True
    )

    # M and B correlation
    MB = pm.MvNormal("MB", mu=tt.stack([muM, muB]), chol=chol, observed=MB_masked)

    mu = a + bB * MB[:, 1] + bM * M

    Ki = pm.Normal("Ki", mu, sigma, observed=K)

    idata_m15_7 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)

idata_m15_7.posterior = idata_m15_7.posterior.rename_vars({"chol_cov_corr": "Rho_MB"})
# Strong correlation between M and B
az.summary(idata_m15_7, var_names=["bM", "bB", "Rho_MB"], round_to=2)


# In[20]:


post = idata_m15_7.posterior

# Calculate the posterior mean and hdi for imputed values of B
B_impute_mu = post["MB_missing"].mean(dim=["chain", "draw"])
B_impute_ci = az.hdi(post, var_names=["MB_missing"])["MB_missing"]

# B vs K
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(B, K)
ax.scatter(B_impute_mu, K[~obs_idx], edgecolor="k", facecolor="none")
ax.plot(
    [B_impute_ci.sel(hdi="lower"), B_impute_ci.sel(hdi="higher")],
    [K[~obs_idx], K[~obs_idx]],
    color="k",
)
ax.set_xlabel("neocortex percent (std)")
ax.set_ylabel("kcal milk (std)")

# M vs B
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(M, B)
ax.scatter(M[~obs_idx], B_impute_mu, edgecolor="k", facecolor="none")
ax.plot(
    [M[~obs_idx], M[~obs_idx]],
    [B_impute_ci.sel(hdi="lower"), B_impute_ci.sel(hdi="higher")],
    color="k",
)
ax.set_xlabel("log body mass (std)")
ax.set_ylabel("neocortex percent (std)");


# #### Code 15.29

# In[21]:


N_houses = 100  # Number of houses to simulate
alpha = 5  # Average number of notes when cat is absent
beta = -3  # Difference in number of notes when cat is present
k_true = 0.5  # Probability of cat present
r = 0.2  # Probability of not knowing whether cat present/absent

cat_true = rng.binomial(n=1, p=k_true, size=N_houses)
notes = rng.poisson(lam=alpha + beta * cat_true, size=N_houses)
R_C = rng.binomial(n=1, p=r, size=N_houses)
cat = cat_true.copy()
cat[R_C == 1] = -9


# #### Code 15.30

# In[22]:


with pm.Model() as m15_8:
    # priors
    a = pm.Normal("a", 0, 1)
    b = pm.Normal("b", 0, 0.5)
    k = pm.Beta("k", 2, 2)

    # cat NA
    custom_logp = pm.math.logsumexp(
        pm.math.log(k)
        + pm.Poisson.dist(pm.math.exp(a + b)).logp(notes[cat == -9])
        + pm.math.log(1 - k)
        + pm.Poisson.dist(pm.math.exp(a)).logp(notes[cat == -9])
    )
    # Using pm.Potential to add custom term to model logp
    notes_RC_1 = pm.Potential("notes|RC==1", custom_logp)

    # cat known present/absent
    cat_RC_0 = pm.Bernoulli("cat|RC==0", k, observed=cat[cat != -9])
    lam = pm.math.exp(a + b * cat_RC_0)
    notes_RC_0 = pm.Poisson("notes|RC==0", lam, observed=notes[cat != -9])

    idata_m15_8 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)
az.summary(idata_m15_8, var_names=["a", "b", "k"], round_to=2)


# In[23]:


print(f"a from data generating process = {round(np.log(alpha), 2)}")
print(f"b from data generating process = {round(np.log(alpha+beta)-np.log(alpha), 2)}")
print(f"k from data generating process = {k_true}")


# In[24]:


az.plot_trace(idata_m15_8, var_names=["a", "b", "k"]);


# #### Code 15.31

# In[25]:


with pm.Model() as m15_9:
    # priors
    a = pm.Normal("a", 0, 1)
    b = pm.Normal("b", 0, 0.5)
    k = pm.Beta("k", 2, 2)

    # cat NA
    custom_logp = pm.math.logsumexp(
        pm.math.log(k)
        + pm.Poisson.dist(pm.math.exp(a + b)).logp(notes[cat == -9])
        + pm.math.log(1 - k)
        + pm.Poisson.dist(pm.math.exp(a)).logp(notes[cat == -9])
    )
    notes_RC_1 = pm.Potential("notes|RC==1", custom_logp)

    # cat known present/absent
    cat_RC_0 = pm.Bernoulli("cat|RC==0", k, observed=cat[cat != -9])
    lam = pm.math.exp(a + b * cat_RC_0)
    notes_RC_0 = pm.Poisson("notes|RC==0", lam, observed=notes[cat != -9])

    # imputed values
    lpC0 = pm.Deterministic(
        "lpC0", pm.math.log(1 - k) + pm.Poisson.dist(pm.math.exp(a)).logp(notes)
    )
    lpC1 = pm.Deterministic(
        "lpC1", pm.math.log(k) + pm.Poisson.dist(pm.math.exp(a + b)).logp(notes)
    )
    PrC1 = pm.Deterministic("PrC1", pm.math.exp(lpC1) / (pm.math.exp(lpC1) + pm.math.exp(lpC0)))

    idata_m15_9 = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)
az.summary(idata_m15_9, var_names=["a", "b", "k"], round_to=2)


# In[26]:


# Posterior P(C==1|N)
PrC1_hdi = az.hdi(idata_m15_9.posterior["PrC1"])["PrC1"]

# For display purposes, sort by whether the cat was absent or present
sorted_cats = np.argsort(cat_true)
cat_true_sorted = cat_true[sorted_cats]
PrC1_hdi_sorted = PrC1_hdi[sorted_cats]
cat_obs_sorted = cat[sorted_cats]

# We will give a different colour to cases where we don't know if the cat is there
cat_obs_sorted[cat_obs_sorted == -9] = 2


# In[27]:


# Plot P(C==1 | N) for each house

labels = ["Absent", "Present", "Unknown"]
colours = ["tab:red", "tab:green", "tab:blue"]

fig, ax = plt.subplots(figsize=(7, 5))
for i in range(3):
    idx = cat_obs_sorted == i
    ax.plot(
        [np.flatnonzero(idx), np.flatnonzero(idx)],
        [PrC1_hdi_sorted[idx, 0], PrC1_hdi_sorted[idx, 1]],
        color=colours[i],
    )
    # Trick to have one label per group
    ax.axhline(2, color=colours[i], alpha=1, label=labels[i])

ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("House")
ax.set_ylabel("P(C=1 | N)")
ax.legend();


# In[28]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-n -u -v -iv -w')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[76]:


import warnings

import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy as sp

from scipy import stats
from scipy.special import expit as logistic
from scipy.special import softmax

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
warnings.simplefilter(action="ignore", category=FutureWarning)
RANDOM_SEED = 8927
np.random.seed(286)


# In[77]:


az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89


def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()


# #### Code 11.1

# In[78]:


d = pd.read_csv("Data/chimpanzees.csv", sep=";")
# we change "actor" to zero-index
d.actor = d.actor - 1
d


# #### Code 11.2

# In[79]:


d["treatment"] = d.prosoc_left + 2 * d.condition
d[["actor", "prosoc_left", "condition", "treatment"]]


# #### Code 11.3

# In[80]:


d.groupby("treatment").first()[["prosoc_left", "condition"]]


# #### Code 11.4 and 11.5

# In[81]:


with pm.Model() as m11_1:
    a = pm.Normal("a", 0.0, 10.0)
    p = pm.Deterministic("p", pm.math.invlogit(a))
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    prior_11_1 = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
idata_11_1 = prior_11_1["prior"]

# need it to recreate the prior pred plot:
with pm.Model() as m11_1bis:
    a = pm.Normal("a", 0.0, 1.5)
    p = pm.Deterministic("p", pm.math.invlogit(a))
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    prior_11_1bis = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
idata_11_1bis = prior_11_1bis["prior"]


# #### Code 11.6

# In[82]:


ax = az.plot_density(
    [idata_11_1, idata_11_1bis],
    data_labels=["a ~ Normal(0, 10)", "a ~ Normal(0, 1.5)"],
    group="prior",
    colors=["k", "b"],
    var_names=["p"],
    point_estimate=None,
)[0]
ax[0].set_xlabel("prior prob pull left")
ax[0].set_ylabel("Density")
ax[0].set_title("Prior predictive simulations for p");


# #### Code 11.7 - 11.9

# In[83]:


with pm.Model() as m11_2:
    a = pm.Normal("a", 0.0, 1.5)
    b = pm.Normal("b", 0.0, 10.0, shape=4)

    p = pm.math.invlogit(a + b[d.treatment.values])
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    prior_11_2 = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
prior_2 = prior_11_2["prior"]

with pm.Model() as m11_3:
    a = pm.Normal("a", 0.0, 1.5)
    b = pm.Normal("b", 0.0, 0.5, shape=4)

    p = pm.math.invlogit(a + b[d.treatment.values])
    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    prior_11_3 = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
prior_3 = prior_11_3["prior"]


# In[84]:


p_treat1, p_treat2 = (
    logistic(prior_2["a"] + prior_2["b"].sel(b_dim_0=0)),
    logistic(prior_2["a"] + prior_2["b"].sel(b_dim_0=1)),
)
p_treat1_bis, p_treat2_bis = (
    logistic(prior_3["a"] + prior_3["b"].sel(b_dim_0=0)),
    logistic(prior_3["a"] + prior_3["b"].sel(b_dim_0=1)),
)

ax = az.plot_density(
    [np.abs(p_treat1 - p_treat2).values, np.abs(p_treat1_bis - p_treat2_bis).values],
    data_labels=["b ~ Normal(0, 10)", "b ~ Normal(0, 0.5)"],
    group="prior",
    colors=["k", "b"],
    point_estimate=None,
)[0]
ax[0].set_xlabel("prior diff between treatments")
ax[0].set_ylabel("Density")
ax[0].set_title(None);


# In[85]:


np.abs(p_treat1_bis - p_treat2_bis).mean().values


# #### Code 11.10

# In[86]:


actor_idx, actors = pd.factorize(d.actor)
treat_idx, treatments = pd.factorize(d.treatment)


# #### Code 11.11

# In[87]:


with pm.Model() as m11_4:
    a = pm.Normal("a", 0.0, 1.5, shape=len(actors))
    b = pm.Normal("b", 0.0, 0.5, shape=len(treatments))

    actor_id = pm.intX(pm.Data("actor_id", actor_idx))
    treat_id = pm.intX(pm.Data("treat_id", treat_idx))
    p = pm.Deterministic("p", pm.math.invlogit(a[actor_id] + b[treat_id]))

    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    trace_11_4 = pm.sample(random_seed=RANDOM_SEED)

az.summary(trace_11_4, var_names=["a", "b"], round_to=2)


# #### Code 11.12

# In[88]:


az.plot_forest(trace_11_4, var_names=["a"], transform=logistic, combined=True);


# #### Code 11.13

# In[89]:


ax = az.plot_forest(trace_11_4, var_names=["b"], combined=True)
ax[0].set_yticklabels(["L/P", "R/P", "L/N", "R/N"]);


# #### Code 11.14

# In[90]:


db13 = trace_11_4.posterior["b"].sel(b_dim_0=0) - trace_11_4.posterior["b"].sel(b_dim_0=2)
db24 = trace_11_4.posterior["b"].sel(b_dim_0=1) - trace_11_4.posterior["b"].sel(b_dim_0=3)
az.plot_forest([db13.values, db24.values], model_names=["db13", "db24"], combined=True);


# #### Code 11.15

# In[91]:


pl = d.groupby(["actor", "treatment"]).agg("mean")["pulled_left"].unstack()
pl


# #### Code 11.16 and 11.17

# In[92]:


with m11_4:
    pm.set_data({"actor_id": np.repeat(range(7), 4), "treat_id": list(range(4)) * 7})
    ppd = pm.sample_posterior_predictive(trace_11_4, random_seed=RANDOM_SEED, var_names=["p"])[
        "posterior_predictive"
    ]["p"]
p_mu = np.array(ppd.mean(["chain", "draw"])).reshape((7, 4))


# In[93]:


pl_val = pl.stack().values
_, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))
alpha, xoff, yoff = 0.6, 0.3, 0.05

ax0.plot([7 * 4 - 0.5] * 2, [0, 1], c="k", alpha=0.4, lw=1)
ax0.axhline(0.5, ls="--", c="k", alpha=0.4)
for actor in range(len(actors)):
    ax0.plot(
        [actor * 4, actor * 4 + 2],
        [pl.loc[actor, 0], pl.loc[actor, 2]],
        "-",
        c="b",
        alpha=alpha,
    )
    ax0.plot(
        [actor * 4 + 1, actor * 4 + 3],
        [pl.loc[actor, 1], pl.loc[actor, 3]],
        "-",
        c="b",
        alpha=alpha,
    )
    ax0.plot(
        [actor * 4, actor * 4 + 1],
        [pl.loc[actor, 0], pl.loc[actor, 1]],
        "o",
        c="b",
        fillstyle="none",
        ms=6,
        alpha=alpha,
    )
    ax0.plot(
        [actor * 4 + 2, actor * 4 + 3],
        [pl.loc[actor, 2], pl.loc[actor, 3]],
        "o",
        c="b",
        ms=6,
        alpha=alpha,
    )
    ax0.plot([actor * 4 - 0.5] * 2, [0, 1], c="k", alpha=0.4, lw=1)
    ax0.text(actor * 4 + 0.5, 1.1, f"actor {actor + 1}", fontsize=12)
    if actor == 0:
        ax0.text(actor * 4 - xoff, pl.loc[actor, 0] - 2 * yoff, "R/N")
        ax0.text(actor * 4 + 1 - xoff, pl.loc[actor, 1] + yoff, "L/N")
        ax0.text(actor * 4 + 2 - xoff, pl.loc[actor, 2] - 2 * yoff, "R/P")
        ax0.text(actor * 4 + 3 - xoff, pl.loc[actor, 3] + yoff, "L/P")
ax0.set_xticks([])
ax0.set_ylabel("proportion left lever", labelpad=10)
ax0.set_title("observed proportions", pad=25)

ax1.plot([range(28), range(28)], az.hdi(ppd)["p"].T, "k-", lw=2, alpha=alpha)
ax1.plot([7 * 4 - 0.5] * 2, [0, 1], c="k", alpha=0.4, lw=1)
ax1.axhline(0.5, ls="--", c="k", alpha=0.4)
for actor in range(len(actors)):
    ax1.plot(
        [actor * 4, actor * 4 + 2],
        [p_mu[actor, 0], p_mu[actor, 2]],
        "-",
        c="k",
        alpha=alpha,
    )
    ax1.plot(
        [actor * 4 + 1, actor * 4 + 3],
        [p_mu[actor, 1], p_mu[actor, 3]],
        "-",
        c="k",
        alpha=alpha,
    )
    ax1.plot(
        [actor * 4, actor * 4 + 1],
        [p_mu[actor, 0], p_mu[actor, 1]],
        "o",
        c="k",
        fillstyle="none",
        ms=6,
        alpha=alpha,
    )
    ax1.plot(
        [actor * 4 + 2, actor * 4 + 3],
        [p_mu[actor, 2], p_mu[actor, 3]],
        "o",
        c="k",
        ms=6,
        alpha=alpha,
    )
    ax1.plot([actor * 4 - 0.5] * 2, [0, 1], c="k", alpha=0.4, lw=1)
    ax1.text(actor * 4 + 0.5, 1.1, f"actor {actor + 1}", fontsize=12)
ax1.set_xticks([])
ax1.set_ylabel("proportion left lever", labelpad=10)
ax1.set_title("posterior predictions", pad=25)


# #### Code 11.18

# In[94]:


side = d.prosoc_left.values  # right 0, left 1
cond = d.condition.values  # no partner 0, partner 1


# #### Code 11.19
with pm.Model() as m11_5:
    a = pm.Normal("a", 0.0, 1.5, shape=len(actors))
    bs = pm.Normal("bs", 0.0, 0.5, shape=2)
    bc = pm.Normal("bc", 0.0, 0.5, shape=2)

    p = pm.math.invlogit(a[actor_idx] + bs[side] + bc[cond])

    pulled_left = pm.Binomial("pulled_left", 1, p, observed=d.pulled_left)

    trace_11_5 = pm.sample(random_seed=RANDOM_SEED)
# #### Code 11.20
# As we changed the data of `m11_4` above, we need to sample from it again, with the original data:

# In[95]:


with m11_4:
    pm.set_data({"actor_id": actor_idx, "treat_id": treat_idx})
    trace_11_4 = pm.sample(random_seed=RANDOM_SEED)

az.compare({"m11_4": trace_11_4, "m11_5": trace_11_5})


# #### Code 11.23

# In[96]:


post_b_11_4 = az.extract_dataset(trace_11_4["posterior"])["b"]
np.exp(np.array(post_b_11_4[3, :]) - np.array(post_b_11_4[1, :])).mean().round(3)


# #### Code 11.24

# In[97]:


d = pd.read_csv("Data/chimpanzees.csv", sep=";")
d.actor = d.actor - 1
d["treatment"] = d.prosoc_left + 2 * d.condition

d_aggregated = (
    d.groupby(["treatment", "actor"]).sum().reset_index()[["treatment", "actor", "pulled_left"]]
)
d_aggregated.head(10)


# #### Code 11.25

# In[98]:


with pm.Model() as m11_6:
    a = pm.Normal("a", 0.0, 1.5, shape=len(actors))
    b = pm.Normal("b", 0.0, 0.5, shape=len(treatments))

    p = pm.Deterministic(
        "p", pm.math.invlogit(a[d_aggregated.actor.values] + b[d_aggregated.treatment.values])
    )

    pulled_left = pm.Binomial("pulled_left", 18, p, observed=d_aggregated.pulled_left)

    trace_11_6 = pm.sample(random_seed=RANDOM_SEED)


# #### Code 11.26
# ArviZ won't even let you compare models with different observations:

# In[99]:


az.compare({"m11_4": trace_11_4, "m11_6": trace_11_6})


# #### Code 11.27

# In[100]:


# deviance of aggregated 6-in-9
(-2 * stats.binom.logpmf(6, 9, 0.2)).round(5)


# In[101]:


# deviance of dis-aggregated
-2 * stats.bernoulli.logpmf([1, 1, 1, 1, 1, 1, 0, 0, 0], 0.2).sum().round(5)


# #### Code 11.28

# In[102]:


d_ad = pd.read_csv("Data/UCBadmit.csv", sep=";")
d_ad


# #### Code 11.29

# In[103]:


gid = (d_ad["applicant.gender"] == "female").astype(int).values

with pm.Model() as m11_7:
    a = pm.Normal("a", 0, 1.5, shape=2)
    p = pm.Deterministic("p", pm.math.invlogit(a[gid]))

    admit = pm.Binomial("admit", p=p, n=d_ad.applications.values, observed=d_ad.admit.values)

    trace_11_7 = pm.sample(random_seed=RANDOM_SEED)
az.summary(trace_11_7, var_names=["a"], round_to=2)


# #### Code 11.30

# In[104]:


post_a_11_7 = az.extract_dataset(trace_11_7["posterior"])["a"]
diff_a = post_a_11_7[0, :] - post_a_11_7[1, :]
diff_p = logistic(post_a_11_7[0, :]) - logistic(post_a_11_7[1, :])
az.summary({"diff_a": diff_a, "diff_p": diff_p}, kind="stats", round_to=2)


# #### Code 11.31

# In[105]:


with m11_7:
    ppc = pm.sample_posterior_predictive(
        trace_11_7, random_seed=RANDOM_SEED, var_names=["admit", "p"]
    )
    pp_p = ppc["posterior_predictive"]["p"]
    pp_admit = ppc["posterior_predictive"]["admit"] / d_ad.applications.values[None, :]


p_mu = np.array(pp_p.mean(["chain", "draw"]))
p_std = np.array(pp_p.std(["chain", "draw"]))
admit_mu = np.array(pp_admit.mean(["chain", "draw"]))
admit_std = np.array(pp_admit.std(["chain", "draw"]))


# In[106]:


for i in range(6):
    x = 1 + 2 * i

    y1 = d_ad.admit[x] / d_ad.applications[x]
    y2 = d_ad.admit[x + 1] / d_ad.applications[x + 1]

    plt.plot([x, x + 1], [y1, y2], "-C0o", alpha=0.6, lw=2)
    plt.text(x + 0.25, (y1 + y2) / 2 + 0.05, d_ad.dept[x])

plt.plot(range(1, 13), p_mu, "ko", fillstyle="none", ms=6, alpha=0.6)
plt.plot([range(1, 13), range(1, 13)], az.hdi(trace_11_7)["p"].T, "k-", lw=1, alpha=0.6)
plt.plot([range(1, 13), range(1, 13)], az.hdi(pp_admit)["admit"].T, "k+", ms=6, alpha=0.6)

plt.xlabel("case")
plt.ylabel("admit")
plt.title("Posterior validation check")
plt.ylim(0, 1);


# #### Code 11.32

# In[107]:


dept_id = pd.Categorical(d_ad["dept"]).codes

with pm.Model() as m11_8:
    a = pm.Normal("a", 0, 1.5, shape=2)
    delta = pm.Normal("delta", 0, 1.5, shape=6)

    p = pm.math.invlogit(a[gid] + delta[dept_id])

    admit = pm.Binomial("admit", p=p, n=d_ad.applications.values, observed=d_ad.admit.values)

    trace_11_8 = pm.sample(2000, random_seed=RANDOM_SEED)
az.summary(trace_11_8, round_to=2)


# #### Code 11.33

# In[108]:


post_a_11_8 = az.extract_dataset(trace_11_8["posterior"])["a"]
diff_a = post_a_11_8[0, :] - post_a_11_8[1, :]
diff_p = logistic(post_a_11_8[0, :]) - logistic(post_a_11_8[1, :])
az.summary({"diff_a": diff_a, "diff_p": diff_p}, kind="stats", round_to=2)


# #### Code 11.34

# In[109]:


pg = pd.DataFrame(index=["male", "female"], columns=d_ad.dept.unique())
for dep in pg.columns:
    pg[dep] = (
        d_ad.loc[d_ad.dept == dep, "applications"]
        / d_ad.loc[d_ad.dept == dep, "applications"].sum()
    ).values
pg.round(2)


# #### Code 11.35

# In[110]:


y = np.random.binomial(n=1000, p=1 / 1000, size=10_000)
y.mean(), y.var()


# #### Code 11.36

# In[111]:


dk = pd.read_csv("Data/Kline", sep=";")
dk


# #### Code 11.37

# In[112]:


P = standardize(np.log(dk.population)).values
c_id = (dk.contact == "high").astype(int).values


# #### Code 11.38

# In[113]:


ax = az.plot_kde(
    np.random.lognormal(0.0, 10.0, size=10_000),
    label="a ~ LogNormal(0, 10)",
    plot_kwargs={"color": "k"},
)
ax.set_xlabel("mean number of tools")
ax.set_ylabel("Density")
ax.set_title("");


# #### Code 11.39

# In[114]:


a = np.random.normal(0.0, 10.0, size=10_000)
np.exp(a).mean()


# #### Code 11.40

# In[115]:


ax = az.plot_kde(np.random.lognormal(3.0, 0.5, size=20_000), label="a ~ LogNormal(3, 0.5)")
ax.set_xlabel("mean number of tools")
ax.set_ylabel("Density")
ax.set_title("");


# #### Code 11.41 to 11.44

# In[116]:


def kline_prior_plot(N: int = 100, b_prior: str = "bespoke", x_scale: str = "stdz", ax=None):
    """
    Utility function to plot prior predictive checks for Kline Poisson model.
    N: number of prior predictive trends.

    """
    if ax is None:
        _, ax = plt.subplots()
    ax.set_ylabel("total tools")

    itcpts = np.random.normal(3.0, 0.5, N)
    if b_prior == "conventional":
        slopes = np.random.normal(0.0, 10.0, N)
        ax.set_title("b ~ Normal(0, 10)")
    elif b_prior == "bespoke":
        slopes = np.random.normal(0.0, 0.2, N)
        ax.set_title("b ~ Normal(0, 0.2)")
    else:
        raise ValueError(
            "Prior for slopes (b_prior) can only be either 'conventional' or 'bespoke'."
        )

    x_seq = np.linspace(np.log(100), np.log(200_000), N)
    ax.set_ylim((0, 500))
    if x_scale == "log":
        for a, b in zip(itcpts, slopes):
            ax.plot(x_seq, np.exp(a + b * x_seq), "k", alpha=0.4)
        ax.set_xlabel("log population")
    elif x_scale == "natural":
        for a, b in zip(itcpts, slopes):
            ax.plot(np.exp(x_seq), np.exp(a + b * x_seq), "k", alpha=0.4)
        ax.set_xlabel("population")
    else:
        x_seq = np.linspace(-2, 2, N)
        for a, b in zip(itcpts, slopes):
            ax.plot(x_seq, np.exp(a + b * x_seq), "k", alpha=0.4)
        ax.set_ylim((0, 100))
        ax.set_xlabel("log population (std)")

    return ax


# In[117]:


_, ax = plt.subplots(2, 2, figsize=(10, 10))
kline_prior_plot(b_prior="conventional", x_scale="stdz", ax=ax[0][0])
kline_prior_plot(b_prior="bespoke", x_scale="stdz", ax=ax[0][1])
kline_prior_plot(b_prior="bespoke", x_scale="log", ax=ax[1][0])
kline_prior_plot(x_scale="natural", ax=ax[1][1])


# #### Code 11.45

# In[118]:


# intercept only
with pm.Model() as m11_9:
    a = pm.Normal("a", 3.0, 0.5)
    T = pm.Poisson("total_tools", pm.math.exp(a), observed=dk.total_tools)
    trace_11_9 = pm.sample(tune=3000, random_seed=RANDOM_SEED)

# interaction model
with pm.Model() as m11_10:
    a = pm.Normal("a", 3.0, 0.5, shape=2)
    b = pm.Normal("b", 0.0, 0.2, shape=2)

    cid = pm.intX(pm.Data("cid", c_id))
    P_ = pm.Data("P", P)
    lam = pm.math.exp(a[cid] + b[cid] * P_)

    T = pm.Poisson("total_tools", lam, observed=dk.total_tools)
    trace_11_10 = pm.sample(tune=3000, random_seed=RANDOM_SEED)


# #### Code 11.46

# In[119]:


az.compare({"m11_9": trace_11_9, "m11_10": trace_11_10})


# In[120]:


# store pareto-k values for plot:
k = az.loo(trace_11_10, pointwise=True).pareto_k.values


# #### Code 11.47 and 11.48

# In[121]:


ns = 100
P_seq = np.linspace(-1.4, 3.0, ns)

with m11_10:
    # predictions for cid=0 (low contact)
    pm.set_data({"cid": np.array([0] * ns), "P": P_seq})
    lam0 = pm.sample_posterior_predictive(trace_11_10, var_names=["total_tools"])[
        "posterior_predictive"
    ]["total_tools"]

    # predictions for cid=1 (high contact)
    pm.set_data({"cid": np.array([1] * ns)})
    lam1 = pm.sample_posterior_predictive(trace_11_10, var_names=["total_tools"])[
        "posterior_predictive"
    ]["total_tools"]

lmu0, lmu1 = lam0.mean(["chain", "draw"]), lam1.mean(["chain", "draw"])


# In[122]:


_, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

# scale point size to Pareto-k:
k /= k.max()
psize = 250 * k

# Plot on standardized log scale:

az.plot_hdi(P_seq, lam1, color="b", fill_kwargs={"alpha": 0.2}, ax=ax0)
ax0.plot(P_seq, lmu1, color="b", alpha=0.7, label="high contact mean")

az.plot_hdi(P_seq, lam0, color="k", fill_kwargs={"alpha": 0.2}, ax=ax0)
ax0.plot(P_seq, lmu0, "--", color="k", alpha=0.7, label="low contact mean")

# display names and k:
mask = k > 0.3
labels = dk.culture.values[mask]
for i, text in enumerate(labels):
    ax0.text(
        P[mask][i] - 0.2,
        dk.total_tools.values[mask][i] + 4,
        f"{text} ({np.round(k[mask][i], 2)})",
        fontsize=8,
    )

# display observed data:
index = c_id == 1
ax0.scatter(
    P[~index],
    dk.total_tools[~index],
    s=psize[~index],
    facecolors="none",
    edgecolors="k",
    alpha=0.8,
    lw=1,
    label="low contact",
)
ax0.scatter(P[index], dk.total_tools[index], s=psize[index], alpha=0.8, label="high contact")
ax0.set_xlabel("log population (std)")
ax0.set_ylabel("total tools")
ax0.legend(fontsize=8, ncol=2)

# Plot on natural scale:
# unstandardize and exponentiate values of standardized log pop:
P_seq = np.linspace(-5.0, 3.0, ns)
P_seq = np.exp(P_seq * np.log(dk.population.values).std() + np.log(dk.population.values).mean())

az.plot_hdi(P_seq, lam1, color="b", fill_kwargs={"alpha": 0.2}, ax=ax1)
ax1.plot(P_seq, lmu1, color="b", alpha=0.7)

az.plot_hdi(P_seq, lam0, color="k", fill_kwargs={"alpha": 0.2}, ax=ax1)
ax1.plot(P_seq, lmu0, "--", color="k", alpha=0.7)

# display observed data:
ax1.scatter(
    dk.population[~index],
    dk.total_tools[~index],
    s=psize[~index],
    facecolors="none",
    edgecolors="k",
    alpha=0.8,
    lw=1,
)
ax1.scatter(dk.population[index], dk.total_tools[index], s=psize[index], alpha=0.8)
plt.setp(ax1.get_xticklabels(), ha="right", rotation=45)
ax1.set_xlim((-10_000, 350_000))
ax1.set_xlabel("population")
ax1.set_ylabel("total tools");


# #### Code 11.49
# The book doesn't pre-process the population data, but if you give them raw to PyMC, the sampler will break: the scale of these data is too wide. However we can't just standardize the data, as we usually do. Why? Because some data points will then be negative, which doesn't play nice with the `b` exponent (try it if you don't trust me). But we'll do something similar: let's standardize the data, and then just add the absolute value of the minimum, and add yet again an epsilon -- this will ensure that our data stay positive and that the transformation will be easy to reverse when we want to plot on the natural scale:

# In[123]:


P = standardize(np.log(dk.population)).values
P = P + np.abs(P.min()) + 0.1  # must be > 0


# And now we can run the model:

# In[124]:


with pm.Model() as m11_11:
    a = pm.Normal("a", 1.0, 1.0, shape=2)
    b = pm.Exponential("b", 1.0, shape=2)
    g = pm.Exponential("g", 1.0)

    cid = pm.intX(pm.Data("cid", c_id))
    P_ = pm.Data("P", P)
    lam = (at.exp(a[cid]) * P_ ** b[cid]) / g

    T = pm.Poisson("total_tools", lam, observed=dk.total_tools.values)
    trace_11_11 = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=RANDOM_SEED)
az.plot_trace(trace_11_11, compact=True);


# #### Bonus: posterior predictive plot for scientific model

# In[125]:


ns = 100
P_seq = np.linspace(-1.4, 3.0, ns) + 1.4  # our little trick

with m11_11:
    # predictions for cid=0 (low contact)
    pm.set_data({"cid": np.array([0] * ns), "P": P_seq})
    lam0 = pm.sample_posterior_predictive(trace_11_11, var_names=["total_tools"])[
        "posterior_predictive"
    ]["total_tools"]

    # predictions for cid=1 (high contact)
    pm.set_data({"cid": np.array([1] * ns)})
    lam1 = pm.sample_posterior_predictive(trace_11_11, var_names=["total_tools"])[
        "posterior_predictive"
    ]["total_tools"]

lmu0, lmu1 = lam0.mean(["chain", "draw"]), lam1.mean(["chain", "draw"])


# In[126]:


_, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

# Plot on standardized log scale:
az.plot_hdi(P_seq, lam1, color="b", fill_kwargs={"alpha": 0.2}, ax=ax0)
ax0.plot(P_seq, lmu1, color="b", alpha=0.7, label="high contact mean")

az.plot_hdi(P_seq, lam0, color="k", fill_kwargs={"alpha": 0.2}, ax=ax0)
ax0.plot(P_seq, lmu0, "--", color="k", alpha=0.7, label="low contact mean")

# display observed data:
index = c_id == 1
ax0.scatter(
    P[~index],
    dk.total_tools[~index],
    s=psize[~index],
    facecolors="none",
    edgecolors="k",
    alpha=0.8,
    lw=1,
    label="low contact",
)
ax0.scatter(P[index], dk.total_tools[index], s=psize[index], alpha=0.8, label="high contact")
ax0.set_xlabel("standardized population")
ax0.set_ylabel("total tools")
ax0.legend(fontsize=8, ncol=2)

# Plot on natural scale:
# unstandardize our log pop sequence:
P_seq = np.exp(
    (P_seq - 1.4) * np.log(dk.population.values).std() + np.log(dk.population.values).mean()
)

az.plot_hdi(P_seq, lam1, color="b", fill_kwargs={"alpha": 0.2}, ax=ax1)
ax1.plot(P_seq, lmu1, color="b", alpha=0.7)

az.plot_hdi(P_seq, lam0, color="k", fill_kwargs={"alpha": 0.2}, ax=ax1)
ax1.plot(P_seq, lmu0, "--", color="k", alpha=0.7)

# display observed data:
ax1.scatter(
    dk.population[~index],
    dk.total_tools[~index],
    s=psize[~index],
    facecolors="none",
    edgecolors="k",
    alpha=0.8,
    lw=1,
)
ax1.scatter(dk.population[index], dk.total_tools[index], s=psize[index], alpha=0.8)
plt.setp(ax1.get_xticklabels(), ha="right", rotation=45)
ax1.set_xlim((-10_000, 350_000))
ax1.set_xlabel("population")
ax1.set_ylabel("total tools");


# #### Code 11.50

# In[127]:


num_days = 30
y = np.random.poisson(1.5, num_days)
y


# #### Code 11.51

# In[128]:


num_weeks = 4
y_new = np.random.poisson(0.5 * 7, num_weeks)
y_new


# #### Code 11.52

# In[129]:


y_all = np.hstack([y, y_new])
exposure = np.hstack([np.repeat(1, 30), np.repeat(7, 4)]).astype("float")
monastery = np.hstack([np.repeat(0, 30), np.repeat(1, 4)])
d = pd.DataFrame({"y": y_all, "days": exposure, "monastery": monastery})
d


# #### Code 11.53

# In[130]:


# compute the offset:
log_days = np.log(exposure)

# fit the model:
with pm.Model() as m11_12:
    a = pm.Normal("a", 0.0, 1.0)
    b = pm.Normal("b", 0.0, 1.0)

    lam = pm.math.exp(log_days + a + b * monastery)

    obs = pm.Poisson("y", lam, observed=y_all)

    trace_11_12 = pm.sample(1000, tune=2000, random_seed=RANDOM_SEED)


# #### Code 11.54

# In[131]:


lambda_old = np.exp(trace_11_12["posterior"]["a"])
lambda_new = np.exp(trace_11_12["posterior"]["a"] + trace_11_12["posterior"]["b"])

az.summary({"lambda_old": lambda_old, "lambda_new": lambda_new}, kind="stats", round_to=2)


# #### Code 11.55

# In[132]:


# simulate career choices among 500 individuals
N = 500  # number of individuals
income = np.array([1, 2, 5])  # expected income of each career
score = 0.5 * income  # score for each career, based on income
# converts scores to probabilities:
p = softmax(score)

# now simulate choice
# outcome career holds event type values, not counts
career = np.random.multinomial(1, p, size=N)
career = np.where(career == 1)[1]
career[:11], score, p


# #### Code 11.56 and 11.57
# 
# The model described in the book does not sample well in PyMC. It does slightly better if we change the pivot category to be the first career instead of the third, but this is still suboptimal because we are discarding predictive information from the pivoted category (i.e., its unique career income). 
# 
# In fact, it is not necessary to pivot the coefficients of variables that are distinct for each category (what the author calls predictors matched to outcomes), as it is done for the coefficients of shared variables (what the author calles predictors matched to observations). The intercepts belong to the second category, and as such they still need to be pivoted. These two references explain this distinction clearly: 
# 
# * Hoffman, S. D., & Duncan, G. J. (1988). Multinomial and conditional logit discrete-choice models in demography. Demography, 25(3), 415-427 [pdf link](https://www.jstor.org/stable/pdf/2061541.pdf)
# * Croissant, Y. (2020). Estimation of Random Utility Models in R: The mlogit Package. Journal of Statistical Software, 95(1), 1-41 [pdf link](https://www.jstatsoft.org/index.php/jss/article/view/v095i11/v95i11.pdf)

# In[133]:


with pm.Model() as m11_13:
    a = pm.Normal("a", 0.0, 1.0, shape=2)  # intercepts
    b = pm.HalfNormal("b", 0.5)  # association of income with choice

    s0 = a[0] + b * income[0]
    s1 = a[1] + b * income[1]
    s2 = 0.0 + b * income[2]  # pivoting the intercept for the third category
    s = pm.math.stack([s0, s1, s2])

    p_ = at.nnet.softmax(s)
    career_obs = pm.Categorical("career", p=p_, observed=career)

    trace_11_13 = pm.sample(tune=2000, target_accept=0.99, random_seed=RANDOM_SEED)
az.summary(trace_11_13, round_to=2)


# #### Code 11.58
# 
# Because this model better matches the actual data generating process, it also does a better job at predicting the effect of doubling the income of the second career.

# In[134]:


# set up logit scores:
post_11_13 = az.extract_dataset(trace_11_13["posterior"])
s0 = post_11_13["a"][0, :] + post_11_13["b"] * income[0]
s1_orig = post_11_13["a"][1, :] + post_11_13["b"] * income[1]
s1_new = post_11_13["a"][1, :] + post_11_13["b"] * income[1] * 2
s2 = 0.0 + post_11_13["b"] * income[2]

pp_scores_orig = np.stack([np.array(s0), np.array(s1_orig), np.array(s2)]).T
pp_scores_new = np.stack([np.array(s0), np.array(s1_new), s2]).T

# compute probabilities for original and counterfactual:
p_orig = softmax(pp_scores_orig, axis=1)
print(p_orig.shape)
p_new = softmax(pp_scores_new, axis=1)
print(p_new.shape)

# summarize
p_diff = p_new[:, 1] - p_orig[:, 1]
az.summary({"p_diff": p_diff}, kind="stats", round_to=2)


# In[135]:


# Actual difference when doubling the income of option #2
income_orig = np.array((1, 2, 5))  # expected income of each career
score_orig = 0.5 * income  # scores for each career, based on income
p_orig = softmax(score_orig)

income_new = np.array((1, 2 * 2, 5))  # expected income of each career
score_new = 0.5 * income_new  # scores for each career, based on income
p_new = softmax(score_new)

print("True p_diff:", p_new[1] - p_orig[1])  # Change in probability of the second career


# #### Code 11.59

# In[136]:


N = 500

# simulate family incomes for each individual
family_income = np.random.rand(N)

# assign a unique coefficient for each type of event
b = np.array([-2.0, 0.0, 2.0])

p = softmax(np.array([0.5, 1.0, 1.5])[:, None] + np.outer(b, family_income), axis=0).T

career = np.asarray([np.random.multinomial(1, pp) for pp in p])
career = np.where(career == 1)[1]
career[:11]


# In[137]:


with pm.Model() as m11_14:
    a = pm.Normal("a", 0.0, 1.5, shape=2)  # intercepts
    b = pm.Normal("b", 0.0, 1.0, shape=2)  # coefficients on family income

    s0 = a[0] + b[0] * family_income
    s1 = a[1] + b[1] * family_income
    s2 = np.zeros(N)  # pivot
    s = pm.math.stack([s0, s1, s2]).T

    p_ = at.nnet.softmax(s)
    career_obs = pm.Categorical("career", p=p_, observed=career)

    trace_11_14 = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=RANDOM_SEED)
az.summary(trace_11_14, round_to=2)


# #### Code 11.60

# In[138]:


d_ad = pd.read_csv("Data/UCBadmit.csv", sep=";")


# #### Code 11.61

# In[139]:


# binomial model of overall admission probability
with pm.Model() as m_binom:
    a = pm.Normal("a", 0, 1.5)
    p = pm.math.invlogit(a)

    admit = pm.Binomial("admit", p=p, n=d_ad.applications.values, observed=d_ad.admit.values)

    trace_binom = pm.sample(1000, tune=2000)

# Poisson model of overall admission and rejection rates
with pm.Model() as m_pois:
    a = pm.Normal("a", 0, 1.5, shape=2)
    lam = pm.math.exp(a)

    admit = pm.Poisson("admit", lam[0], observed=d_ad.admit.values)
    rej = pm.Poisson("rej", lam[1], observed=d_ad.reject.values)

    trace_pois = pm.sample(1000, tune=2000)


# #### Code 11.62

# In[140]:


m_binom = az.summary(trace_binom)
logistic(m_binom["mean"]).round(3)


# #### Code 11.63

# In[141]:


m_pois = az.summary(trace_pois).round(2)
(np.exp(m_pois["mean"][0]) / (np.exp(m_pois["mean"][0]) + np.exp(m_pois["mean"][1]))).round(3)


# In[142]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-n -u -v -iv -w')


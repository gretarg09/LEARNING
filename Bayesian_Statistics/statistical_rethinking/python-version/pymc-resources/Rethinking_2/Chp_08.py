#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from scipy import stats

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


get_ipython().run_line_magic('config', "Inline.figure_format = 'retina'")
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89  # set credible interval for entire notebook
az.rcParams["stats.information_criterion"] = "waic"  # set information criterion to use in `compare`
az.rcParams["stats.ic_scale"] = "deviance"  # set information criterion scale
np.random.seed(0)


# #### Code 8.1

# In[3]:


d = pd.read_csv("Data/rugged.csv", delimiter=";")

# make log version of the outcome
d["log_gdp"] = np.log(d["rgdppc_2000"])

# extract countries with GDP data
dd = d.dropna(subset=["log_gdp"])

# rescale variables
dd["log_gdp_std"] = dd["log_gdp"] / dd["log_gdp"].mean()
dd["rugged_std"] = dd["rugged"] / dd["rugged"].max()


# In[4]:


fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

sns.regplot(
    dd.loc[dd["cont_africa"] == 1]["rugged_std"],
    dd.loc[dd["cont_africa"] == 1]["log_gdp_std"],
    scatter_kws={"color": "b"},
    line_kws={"color": "k"},
    ax=axs[0],
)
sns.regplot(
    dd.loc[dd["cont_africa"] == 0]["rugged_std"],
    dd.loc[dd["cont_africa"] == 0]["log_gdp_std"],
    scatter_kws={"edgecolor": "k", "facecolor": "w"},
    line_kws={"color": "k"},
    ax=axs[1],
)

axs[0].set_ylabel("log GDP (as proportion of mean)")
axs[1].set_ylabel("")
axs[0].set_title("African nations")
axs[1].set_title("Non-African nations")

# label countries
for _, africa in dd.loc[(dd["rugged_std"] > 0.7) & (dd["cont_africa"] == 1)].iterrows():
    axs[0].text(
        africa["rugged_std"],
        africa["log_gdp_std"] - 0.02,
        africa["country"],
        ha="center",
    )

for _, non_africa in dd.loc[(dd["rugged_std"] > 0.7) & (dd["cont_africa"] == 0)].iterrows():
    axs[1].text(
        non_africa["rugged_std"] + 0.03,
        non_africa["log_gdp_std"],
        non_africa["country"],
        va="center",
    )

for ax in axs:
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("ruggedness (standardised)")


# #### Code 8.2

# In[5]:


with pm.Model() as m_8_1:
    a = pm.Normal("a", 1, 1)
    b = pm.Normal("b", 0, 1)

    mu = a + b * (dd["rugged_std"].values - 0.215)
    sigma = pm.Exponential("sigma", 1)

    log_gdp_std = pm.Normal("log_gdp_std", mu, sigma, shape=dd.shape[0])


# #### Code 8.3

# In[6]:


with m_8_1:
    m_8_1_prior = pm.sample_prior_predictive()

# Figure 8.3 is below


# #### Code 8.4

# In[7]:


beta_prior = az.extract_dataset(m_8_1_prior.prior)["b"].to_numpy()
np.sum(np.abs(beta_prior > 0.6)) / len(beta_prior)


# #### Code 8.5

# In[8]:


with pm.Model() as m_8_1t:
    a = pm.Normal("a", 1, 0.1)
    b = pm.Normal("b", 0, 0.3)

    mu = a + b * (dd["rugged_std"].values - 0.215)
    sigma = pm.Exponential("sigma", 1)

    log_gdp_std = pm.Normal("log_gdp_std", mu, sigma, observed=dd["log_gdp_std"].values)

    m_8_1t_trace = pm.sample()

    m_8_1t_prior = pm.sample_prior_predictive()


# In[9]:


# Figure 8.3

_, (ax1, ax2) = plt.subplots(1, 2, figsize=[7, 4], constrained_layout=True)

n = 100
rugged_plot = np.linspace(-0.1, 1.1, n)

# Prior 1
prior = m_8_1_prior.prior.sel(draw=slice(None, None, int(len(m_8_1_prior.prior.draw) / n)))
reglines = prior["a"].T.to_numpy() + rugged_plot * prior["b"].T.to_numpy()
for regline in reglines:
    ax1.plot(
        rugged_plot,
        regline,
        color="k",
        lw=1,
        alpha=0.3,
    )
ax1.set_title("a ~ Normal(1, 1)\nb ~ Normal(0, 1)")

# Prior 2
prior_t = m_8_1t_prior.prior.sel(draw=slice(None, None, int(len(m_8_1t_prior.prior.draw) / n)))
reglines_t = prior_t["a"].T.to_numpy() + rugged_plot * prior_t["b"].T.to_numpy()

for regline in reglines_t:
    ax2.plot(
        rugged_plot,
        regline,
        color="k",
        lw=1,
        alpha=0.3,
    )
ax2.set_title("a ~ Normal(1, 0.1)\nb ~ Normal(0, 0.3)")

for ax in (ax1, ax2):
    ax.set_xlabel("ruggedness")
    ax.set_ylabel("log GDP (prop of mean)")
    ax.axhline(0.7, ls="dashed", color="k", lw=1)
    ax.axhline(1.3, ls="dashed", color="k", lw=1)
    ax.set_ylim(0.5, 1.5)


# #### Code 8.6

# In[10]:


az.summary(m_8_1t_trace, kind="stats", round_to=2)


# #### Code 8.7

# In[11]:


cid = pd.Categorical(dd["cont_africa"])


# #### Code 8.8

# In[12]:


with pm.Model() as m_8_2:
    a = pm.Normal("a", 1, 0.1, shape=cid.categories.size)
    b = pm.Normal("b", 0, 0.3)

    mu = a[np.array(cid)] + b * (dd["rugged_std"].values - 0.215)
    sigma = pm.Exponential("sigma", 1)

    log_gdp_std = pm.Normal("log_gdp_std", mu, sigma, observed=dd["log_gdp_std"].values)

    m_8_2_trace = pm.sample()


# #### Code 8.9

# In[13]:


az.compare({"m_8_1t": m_8_1t_trace, "m_8_2": m_8_2_trace})


# #### Code 8.10

# In[14]:


az.summary(m_8_2_trace, kind="stats", round_to=2)


# #### Code 8.11

# In[15]:


m_8_2_posterior = az.extract_dataset(m_8_2_trace.posterior)
diff_a0_a1 = m_8_2_posterior["a"][1, :] - m_8_2_posterior["a"][0, :]
az.hdi(diff_a0_a1.to_numpy())


# #### Code 8.12

# In[16]:


fig, ax = plt.subplots()

# extract posterior samples of parameters
a0 = m_8_2_posterior["a"][0, :].to_numpy()
a1 = m_8_2_posterior["a"][1, :].to_numpy()
b = m_8_2_posterior["b"].to_numpy()

rugged_plot = np.linspace(-0.1, 1.1)

ax.scatter(
    dd.loc[cid == 0, "rugged_std"],
    dd.loc[cid == 0, "log_gdp_std"],
    label="Not Africa",
    facecolor="w",
    lw=1,
    edgecolor="k",
)
pred0 = a0 + rugged_plot.reshape(-1, 1) * b
ax.plot(rugged_plot, pred0.mean(1), color="grey")
az.plot_hdi(rugged_plot, pred0.T, color="grey", hdi_prob=0.97, ax=ax)

ax.scatter(
    dd.loc[cid == 1, "rugged_std"],
    dd.loc[cid == 1, "log_gdp_std"],
    label="Africa",
    color="b",
)
pred1 = a1 + rugged_plot.reshape(-1, 1) * b
ax.plot(rugged_plot, pred1.mean(1), color="b")
az.plot_hdi(rugged_plot, pred1.T, color="b", hdi_prob=0.97, ax=ax, fill_kwargs={"alpha": 0.2})

ax.legend(frameon=True)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0.7, 1.3)
ax.set_xlabel("ruggedness (standardised)")
ax.set_ylabel("log GDP (as proportion of mean)");


# #### Code 8.13

# In[17]:


with pm.Model() as m_8_3:
    a = pm.Normal("a", 1, 0.1, shape=cid.categories.size)
    b = pm.Normal("b", 0, 0.3, shape=cid.categories.size)

    mu = a[np.array(cid)] + b[np.array(cid)] * (dd["rugged_std"].values - 0.215)
    sigma = pm.Exponential("sigma", 1)

    log_gdp_std = pm.Normal("log_gdp_std", mu, sigma, observed=dd["log_gdp_std"].values)

    m_8_3_trace = pm.sample()


# #### Code 8.14

# In[18]:


az.summary(m_8_3_trace, kind="stats", round_to=2)


# #### Code 8.15

# In[19]:


az.compare({"m_8_1t": m_8_1t_trace, "m_8_2": m_8_2_trace, "m_8_3": m_8_3_trace}, ic="loo")


# #### Code 8.16

# In[20]:


m_8_3_loo = az.loo(m_8_3_trace, pointwise=True)

plt.plot(m_8_3_loo.loo_i)


# #### Code 8.17

# In[21]:


m_8_3_posterior = az.extract_dataset(m_8_3_trace.posterior)


# In[22]:


_, axs = plt.subplots(
    1,
    2,
    figsize=[8, 4],
    sharey=True,
    constrained_layout=True,
)

ax1, ax0 = axs

# extract posterior samples of parameters
a0 = m_8_3_posterior["a"][0, :].to_numpy()
a1 = m_8_3_posterior["a"][1, :].to_numpy()
b0 = m_8_3_posterior["b"][0, :].to_numpy()
b1 = m_8_3_posterior["b"][1, :].to_numpy()

rugged_plot = np.linspace(-0.1, 1.1)

ax0.scatter(
    dd.loc[cid == 0, "rugged_std"],
    dd.loc[cid == 0, "log_gdp_std"],
    label="Not Africa",
    facecolor="w",
    lw=1,
    edgecolor="k",
)
# calculating predicted manually because this is a pain with categorical variabiles in PyMC
pred0 = a0 + rugged_plot.reshape(-1, 1) * b0
ax0.plot(rugged_plot, pred0.mean(1), color="grey")
az.plot_hdi(rugged_plot, pred0.T, color="grey", hdi_prob=0.97, ax=ax0)
ax0.set_title("Non-African Nations")

ax1.scatter(
    dd.loc[cid == 1, "rugged_std"],
    dd.loc[cid == 1, "log_gdp_std"],
    label="Africa",
    color="b",
)
# calculating predicted manually because this is a pain with categorical variabiles in PyMC
pred1 = a1 + rugged_plot.reshape(-1, 1) * b1
ax1.plot(rugged_plot, pred1.mean(1), color="k")
az.plot_hdi(
    rugged_plot,
    pred1.T,
    color="blue",
    hdi_prob=0.97,
    ax=ax1,
    fill_kwargs={"alpha": 0.2},
)
ax1.set_title("African Nations")


ax.set_xlim(-0.1, 1.1)
ax0.set_xlabel("ruggedness (standardised)")
ax1.set_xlabel("ruggedness (standardised)")
ax0.set_ylabel("")
ax1.set_ylabel("log GDP (as proportion of mean)")


# #### Code 8.18

# In[23]:


fig, ax = plt.subplots(figsize=(6, 5))

rugged_plot = np.linspace(-0.1, 1.1)

delta = pred1 - pred0  # using 'pred' from above

ax.plot(rugged_plot, delta.mean(1), c="k")
az.plot_hdi(rugged_plot, delta.T, ax=ax, color="grey")

ax.axhline(0, ls="dashed", zorder=1, color=(0, 0, 0, 0.5))
ax.text(0.01, 0.01, "Africa higher GDP")
ax.text(0.01, -0.03, "Africa lower GDP")

ax.set_xlabel("ruggedness")
ax.set_ylabel("expected difference log GDP")
ax.set_xlim(0, 1)


# These numbers are quite different from the book - not sure why.

# #### Code 8.19

# In[24]:


d = pd.read_csv("Data/tulips.csv", delimiter=";")
d.head()


# $$B_i \sim Normal(\mu_i,\sigma)$$
# 
# $$\mu_i=\alpha+\beta_W(W_i−\overline{W})+\beta_S(S_i−\overline{S})$$

# #### Code 8.20

# In[25]:


d["blooms_std"] = d["blooms"] / d["blooms"].max()
d["water_cent"] = d["water"] - d["water"].mean()
d["shade_cent"] = d["shade"] - d["shade"].mean()


# #### Code 8.21

# In[26]:


a = stats.norm.rvs(0.5, 1, 10000)
sum((a < 0) | (a > 1)) / len(a)


# #### Code 8.22

# In[27]:


a = stats.norm.rvs(0.5, 0.25, 10000)
sum((a < 0) | (a > 1)) / len(a)


# #### Code 8.23

# In[28]:


with pm.Model() as m_8_4:
    a = pm.Normal("a", 0.5, 0.25)
    bw = pm.Normal("bw", 0, 0.25)
    bs = pm.Normal("bs", 0, 0.25)

    mu = a + bw * d["water_cent"].values + bs * d["shade_cent"].values
    sigma = pm.Exponential("sigma", 1)

    blooms_std = pm.Normal("blooms_std", mu, sigma, observed=d["blooms_std"].values)

    m_8_4_trace = pm.sample()
    m_8_4_post = az.extract_dataset(m_8_4_trace.posterior)


# $$B_i \sim Normal(\mu_i,\sigma)$$
# 
# $$\mu_i=\alpha+\beta_W(W_i−\overline{W})+\beta_S(S_i−\overline{S}) +\beta_{WS}W_iS_i $$

# #### Code 8.24

# In[29]:


with pm.Model() as m_8_5:
    a = pm.Normal("a", 0.5, 0.25)
    bw = pm.Normal("bw", 0, 0.25)
    bs = pm.Normal("bs", 0, 0.25)
    bws = pm.Normal("bws", 0, 0.25)

    mu = (
        a
        + bw * d["water_cent"].values
        + bs * d["shade_cent"].values
        + bws * d["water_cent"].values * d["shade_cent"].values
    )
    sigma = pm.Exponential("sigma", 1)

    blooms_std = pm.Normal("blooms_std", mu, sigma, observed=d["blooms_std"].values)

    m_8_5_trace = pm.sample()
    m_8_5_post = az.extract_dataset(m_8_5_trace.posterior)


# #### Code 8.25

# In[30]:


_, axs = plt.subplots(2, 3, figsize=[9, 5], sharey=True, sharex=True, constrained_layout=True)

n_lines = 20
pred_x = np.array([-1, 1])

for i, shade in enumerate([-1, 0, 1]):
    ind = d.shade_cent == shade
    for ax in axs[:, i]:
        ax.scatter(d.loc[ind, "water_cent"], d.loc[ind, "blooms_std"])
    # top row, m_8_4
    ax = axs[0, i]
    ax.set_title(f"m8.4 post: shade = {shade:.0f}", fontsize=11)
    pred_y = (
        m_8_4_post["a"][:n_lines].to_numpy()
        + m_8_4_post["bw"][:n_lines].to_numpy() * pred_x.reshape(-1, 1)
        + m_8_4_post["bs"][:n_lines].to_numpy() * shade
    )
    ax.plot(pred_x, pred_y, lw=1, color=(0, 0, 0, 0.4))

    # bottom row, m_8_5
    ax = axs[1, i]
    ax.set_title(f"m8.5 post: shade = {shade:.0f}", fontsize=11)
    pred_y = (
        m_8_5_post["a"][:n_lines].to_numpy()
        + m_8_5_post["bw"][:n_lines].to_numpy() * pred_x.reshape(-1, 1)
        + m_8_5_post["bs"][:n_lines].to_numpy() * shade
        + m_8_5_post["bws"][:n_lines].to_numpy() * pred_x.reshape(-1, 1) * shade
    )
    ax.plot(pred_x, pred_y, lw=1, color=(0, 0, 0, 0.4))

for ax in axs.flat:
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("blooms")
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel("water");


# #### Code 8.26

# In[31]:


with m_8_4:
    m_8_4_priors = pm.sample_prior_predictive(var_names=["a", "bw", "bs"])
    m_8_4_priors = az.extract_dataset(m_8_4_priors.prior)

with m_8_5:
    m_8_5_priors = pm.sample_prior_predictive(var_names=["a", "bw", "bs", "bws"])
    m_8_5_priors = az.extract_dataset(m_8_5_priors.prior)


# In[32]:


_, axs = plt.subplots(2, 3, figsize=[9, 5], sharey=True, sharex=True, constrained_layout=True)

n_lines = 20
pred_x = np.array([-1, 1])

for i, shade in enumerate([-1, 0, 1]):
    # top row, m_8_4
    ax = axs[0, i]
    ax.set_title(f"m8.4 prior: shade = {shade:.0f}", fontsize=11)
    pred_y = (
        m_8_4_priors["a"][:n_lines].to_numpy()
        + m_8_4_priors["bw"][:n_lines].to_numpy() * pred_x.reshape(-1, 1)
        + m_8_4_priors["bs"][:n_lines].to_numpy() * shade
    )
    ax.plot(pred_x, pred_y, lw=1, color=(0, 0, 0, 0.4))
    ax.plot(pred_x, pred_y[:, 0], lw=2, color="k")

    # bottom row, m_8_5
    ax = axs[1, i]
    ax.set_title(f"m8.5 prior: shade = {shade:.0f}", fontsize=11)
    pred_y = (
        m_8_5_priors["a"][:n_lines].to_numpy()
        + m_8_5_priors["bw"][:n_lines].to_numpy() * pred_x.reshape(-1, 1)
        + m_8_5_priors["bs"][:n_lines].to_numpy() * shade
        + m_8_5_priors["bws"][:n_lines].to_numpy() * pred_x.reshape(-1, 1) * shade
    )
    ax.plot(pred_x, pred_y, lw=1, color=(0, 0, 0, 0.4))
    ax.plot(pred_x, pred_y[:, 0], lw=2, color="k")

for ax in axs.flat:
    ax.set_ylim(-0.5, 1.5)
    ax.axhline(1, ls="dashed", color=(0, 0, 0, 0.6))
    ax.axhline(0, ls="dashed", color=(0, 0, 0, 0.6))
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("blooms")
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel("water");


# In[34]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-n -u -v -iv -w')


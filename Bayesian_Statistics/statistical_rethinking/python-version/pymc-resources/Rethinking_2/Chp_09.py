#!/usr/bin/env python
# coding: utf-8

# # Chapter 9: Markov Chain Monte Carlo
# 
# Statistical Rethinking, 2nd Edition

# In[1]:


import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("mode.chained_assignment", None)


# In[2]:


get_ipython().run_line_magic('config', "Inline.figure_format = 'retina'")
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89  # sets default credible interval used by arviz


# #### Code 9.1

# In[3]:


num_weeks = int(1e5)
positions = np.zeros(num_weeks)
current = 10
for i in range(num_weeks):
    # record current position
    positions[i] = current
    # flip coin to generate proposal
    proposal = current + np.random.choice([-1, 1])
    # loop around the archipelago
    if proposal < 1:
        proposal = 10
    if proposal > 10:
        proposal = 1
    # move?
    prob_move = proposal / current
    if np.random.uniform() < prob_move:
        current = proposal


# #### Code 9.2 & 9.3

# In[4]:


# Figure 9.2
_, axs = plt.subplots(1, 2, figsize=[8, 3.5], constrained_layout=True)

ax0, ax1 = axs

nplot = 100
ax0.scatter(range(nplot), positions[:nplot], s=10)
ax0.set_ylabel("island")
ax0.set_xlabel("week")

counts, _ = np.histogram(positions, bins=10)
ax1.bar(range(10), counts, width=0.1)
ax1.set_ylabel("number of weeks")
ax1.set_xlabel("island")
ax1.set_xticks(range(10))
ax1.set_xticklabels(range(1, 11));


# #### Code 9.4

# In[5]:


def rad_dist(Y):
    return np.sqrt(np.sum(Y**2))


fig, ax = plt.subplots(1, 1, figsize=[7, 3])
xvar = np.linspace(0, 36, 200)

# the book code is wrapped in a loop to reproduce Figure 9.4
for D in [1, 10, 100, 1000]:
    T = int(1e3)
    Y = stats.multivariate_normal(np.zeros(D), np.identity(D)).rvs(T)

    Rd = list(map(rad_dist, Y))

    kde = stats.gaussian_kde(Rd)
    yvar = kde(xvar)
    ax.plot(xvar, yvar, color="k")

    ax.text(np.mean(Rd), np.max(yvar) * 1.02, f"D: {D}")

ax.set_xlim(0, 36)
ax.set_xlabel("Radial distance from mode")
ax.set_ylabel("Density");


# #### Code 9.5

# In[6]:


def calc_U(x, y, q, a=0, b=1, k=0, d=1):
    muy, mux = q

    U = (
        np.sum(stats.norm.logpdf(y, loc=muy, scale=1))
        + np.sum(stats.norm.logpdf(x, loc=mux, scale=1))
        + stats.norm.logpdf(muy, loc=a, scale=b)
        + stats.norm.logpdf(mux, loc=k, scale=d)
    )

    return -U


# #### Code 9.6

# In[7]:


# gradient function
# need vector of partial derivatives of U with respect to vector q
def calc_U_gradient(x, y, q, a=0, b=1, k=0, d=1):
    muy, mux = q

    G1 = np.sum(y - muy) + (a - muy) / b**2  # dU/dmuy
    G2 = np.sum(x - mux) + (k - mux) / b**2  # dU/dmux

    return np.array([-G1, -G2])


# #### Code 9.8 - 9.10
# 
# 9.7 uses the function defined here, so is below

# In[8]:


def HMC2(U, grad_U, epsilon, L, current_q, x, y):
    q = current_q
    p = np.random.normal(loc=0, scale=1, size=len(q))  # random flick - p is momentum
    current_p = p
    # Make a half step for momentum at the beginning
    p -= epsilon * grad_U(x, y, q) / 2
    # initialize bookkeeping - saves trajectory
    qtraj = np.full((L + 1, len(q)), np.nan)
    ptraj = qtraj.copy()
    qtraj[0, :] = current_q
    ptraj[0, :] = p

    # Code 9.9 starts here
    # Alternate full steps for position and momentum
    for i in range(L):
        q += epsilon * p  # Full step for the position
        qtraj[i + 1, :] = q
        # Make a full step for the momentum, except at the end of trajectory
        if i != L - 1:
            p -= epsilon * grad_U(x, y, q)
            ptraj[i + 1, :] = p

    # Make a half step for momentum at the end
    p -= epsilon * grad_U(x, y, q) / 2
    ptraj[L, :] = p
    # Negate momentum at end of trajectory to make the proposal symmetric
    p *= -1
    # Evaluate potential and kinetic energies sat start and end of trajectory
    current_U = U(x, y, current_q)
    current_K = np.sum(current_p**2) / 2
    proposed_U = U(x, y, q)
    proposed_K = np.sum(p**2) / 2
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    accept = False
    if np.random.uniform() < np.exp(current_U - proposed_U + current_U - proposed_K):
        new_q = q  # accept
        accept = True
    else:
        new_q = current_q  # reject

    return dict(q=new_q, traj=qtraj, ptraj=ptraj, accept=accept)


# #### Code 9.7
# 
# This code expands upon 9.7 to produce both panels in the top row of Figure 9.6

# In[9]:


np.random.seed(42)

# test data
real = stats.multivariate_normal([0, 0], np.identity(2))
x, y = real.rvs(50).T

Q = {}
Q["q"] = np.array([-0.1, 0.2])
pr = 0.5
step = 0.03
# L = 11  # 0.03 / 28 for U-turns -- 11 for working example
n_samples = 4

_, axs = plt.subplots(1, 2, figsize=[8, 6], constrained_layout=True)

for L, ax in zip([11, 28], axs):
    ax.scatter(*Q["q"], color="k", marker="x", zorder=3)
    if L == 11:
        ax.text(*Q["q"] + 0.015, "start", weight="bold")
    for i in range(n_samples):
        Q = HMC2(calc_U, calc_U_gradient, step, L, Q["q"], x, y)
        ax.scatter(*Q["q"], color="w", marker="o", edgecolor="k", lw=2, zorder=3)
        if n_samples < 10:
            for j in range(L):
                K0 = np.sum(Q["ptraj"][j, :] ** 2) / 2  # kinetic energy
                ax.plot(
                    Q["traj"][j : j + 2, 0],
                    Q["traj"][j : j + 2, 1],
                    color="k",
                    lw=1 + 1 * K0,
                    alpha=0.3,
                    zorder=1,
                )
            ax.scatter(*Q["traj"].T, facecolor="w", edgecolor="gray", lw=1, zorder=2, s=10)
            if L == 11:
                ax.text(*Q["q"] + [0.02, -0.03], f"{i + 1}", weight="bold")

    ax.set_title(f"2D Gaussian, L = {L}")
    ax.set_xlabel("mux")
    ax.set_ylabel("muy")

    # draw background contours based on real probability defined above
    ax.set_xlim(-pr, pr)
    ax.set_ylim(-pr, pr)
    xs, ys = np.mgrid[-pr:pr:0.01, -pr:pr:0.01]
    p = real.logpdf(np.vstack([xs.flat, ys.flat]).T).reshape(xs.shape)
    ax.contour(xs, ys, p, 4, colors=[(0, 0, 0, 0.3)])
    ax.set_aspect(1);


# In[10]:


# Figure 9.6 (bottom row) with correlated data

np.random.seed()

# test data
realc = stats.multivariate_normal([0, 0], [[1, -0.9], [-0.9, 1]])  # generate correlated data
x, y = real.rvs(50).T

Q = {}
pr = 0.6
step = 0.03
L = 21  # 28 for U-turns -- 11 for working example

_, axs = plt.subplots(1, 2, figsize=[8, 4], constrained_layout=True)

for n_samples, ax in zip([4, 50], axs):
    Q["q"] = np.array([-0.3, 0.3])

    ax.scatter(*Q["q"], color="k", marker="x", zorder=3)
    if n_samples == 4:
        ax.text(*Q["q"] + 0.015, "start", weight="bold")
    for i in range(n_samples):
        Q = HMC2(calc_U, calc_U_gradient, step, L, Q["q"], x, y)
        ax.scatter(*Q["q"], color="w", marker="o", edgecolor="k", lw=2, zorder=3)
        if n_samples < 10:
            for j in range(L):
                K0 = np.sum(Q["ptraj"][j, :] ** 2) / 2  # kinetic energy
                ax.plot(
                    Q["traj"][j : j + 2, 0],
                    Q["traj"][j : j + 2, 1],
                    color="k",
                    lw=1 + 1 * K0,
                    alpha=0.3,
                    zorder=1,
                )
            if Q["accept"]:
                c = "w"
            else:
                c = "k"
            ax.scatter(*Q["traj"].T, facecolor=c, edgecolor="gray", lw=1, zorder=2, s=10)
            if n_samples == 4:
                ax.text(*Q["q"] + [0.02, -0.03], f"{i + 1}", weight="bold")

    ax.set_title(f"Correlation -0.9")
    ax.set_xlabel("mux")
    ax.set_ylabel("muy")

    # draw background contours based on real probability defined above
    ax.set_xlim(-pr, pr)
    ax.set_ylim(-pr, pr)
    xs, ys = np.mgrid[-pr:pr:0.01, -pr:pr:0.01]
    p = realc.logpdf(np.vstack([xs.flat, ys.flat]).T).reshape(xs.shape)
    ax.contour(xs, ys, p, 5, colors=[(0, 0, 0, 0.3)])
    ax.set_aspect(1);


# The correlated case doens't work as well as the uncorrelated case, compared to the book. We're not reproducing the sinusoid trajectories that are shown in the book, so it's possible that there was some underlying change in the likelihood functions?The book doesn't provide code or parameters used to recreate these plots, so we're guessing blind here.

# #### Code 9.11

# In[11]:


d = pd.read_csv("Data/rugged.csv", delimiter=";")

d["log_gdp"] = np.log(d["rgdppc_2000"])

dd = d.dropna(subset=["log_gdp"])
dd["log_gdp_std"] = dd["log_gdp"] / dd["log_gdp"].mean()
dd["rugged_std"] = dd["rugged"] / dd["rugged"].max()


# #### Code 9.12-9.18
# 
# By using PyMC we are already doing everything in these code blocks (No-Uturn sampling, parallell processing).
# 
# To translate the results of `summary` to `rethinking`'s `precis`:
# 
#  - `n_eff = ess_mean`
#  - `Rhat4 = r_hat`

# In[12]:


cid = pd.Categorical(dd["cont_africa"])

with pm.Model() as m_8_3:
    a = pm.Normal("a", 1, 0.2, shape=cid.categories.size)
    b = pm.Normal("b", 0, 0.3, shape=cid.categories.size)

    mu = a[np.array(cid)] + b[np.array(cid)] * (dd["rugged_std"].values - 0.215)
    sigma = pm.Exponential("sigma", 1)

    log_gdp_std = pm.Normal("log_gdp_std", mu, sigma, observed=dd["log_gdp_std"].values)

    m_8_3_trace = pm.sample()

az.summary(m_8_3_trace, kind="all", round_to=2)


# #### Code 9.19

# In[13]:


az.plot_pair(m_8_3_trace, figsize=[8, 8], marginals=True);


# #### Code 9.20

# In[14]:


az.plot_trace(m_8_3_trace, figsize=[8, 8]);


# #### Code 9.21

# In[15]:


az.plot_trace(m_8_3_trace, figsize=[8, 8], kind="rank_bars");


# #### Code 9.22

# In[16]:


y = np.array([-1, 1])

with pm.Model() as m_9_2:
    alpha = pm.Normal("alpha", 0, 1000)

    mu = alpha
    sigma = pm.Exponential("sigma", 0.0001)

    yp = pm.Normal("y", mu, sigma, observed=y)

    m_9_2_trace = pm.sample(chains=3)


# #### Code 9.23

# In[17]:


az.summary(m_9_2_trace, round_to=2)


# In[18]:


# Figure 9.9 (top)
az.plot_trace(m_9_2_trace, figsize=[8, 3]);


# #### Code 9.24

# In[19]:


with pm.Model() as m_9_3:
    alpha = pm.Normal("alpha", 1, 10)

    mu = alpha
    sigma = pm.Exponential("sigma", 1)

    yp = pm.Normal("y", mu, sigma, observed=y)

    m_9_3_trace = pm.sample(chains=3)

az.summary(m_9_3_trace, round_to=2)


# In[20]:


# Figure 9.9 (bottom)
az.plot_trace(m_9_3_trace, figsize=[8, 3]);


# In[21]:


# Figure 9.10
with m_9_3:
    m_9_3_prior = az.extract_dataset(
        pm.sample_prior_predictive(var_names=["alpha", "sigma"])["prior"]
    )
    m_9_3_post = az.extract_dataset(m_9_3_trace["posterior"], var_names=["alpha", "sigma"])

_, axs = plt.subplots(1, 2, figsize=[8, 3.5], constrained_layout=True)
ax0, ax1 = axs

az.plot_kde(m_9_3_prior["alpha"].to_numpy(), ax=ax0, plot_kwargs={"color": "k", "ls": "dashed"})
az.plot_kde(m_9_3_post["alpha"].to_numpy(), ax=ax0, plot_kwargs={"color": "k"})
ax0.set_xlim(-15, 15)
ax0.set_xlabel("alpha")

az.plot_kde(
    m_9_3_prior["sigma"].to_numpy(),
    ax=ax1,
    plot_kwargs={"color": "k", "ls": "dashed"},
    label="prior",
)
az.plot_kde(m_9_3_post["sigma"].to_numpy(), ax=ax1, plot_kwargs={"color": "k"}, label="posterior")
ax1.legend()
ax1.set_xlim(0, 10)
ax1.set_xlabel("sigma")

for ax in axs:
    ax.set_ylabel("Density");


# #### Code 9.25

# In[22]:


y = np.random.normal(loc=0, scale=1, size=100)


# #### Code 9.26

# In[23]:


with pm.Model() as m_9_4:
    a1 = pm.Normal("a1", 0, 1000)
    a2 = pm.Normal("a2", 0, 1000)

    mu = a1 + a2
    sigma = pm.Exponential("sigma", 1)

    yp = pm.Normal("y", mu, sigma, observed=y)

    m_9_4_trace = pm.sample(chains=3)

az.summary(m_9_4_trace, round_to=2)


# In[24]:


# Figure 9.11 (top half)
az.plot_trace(m_9_4_trace, figsize=[8, 4.5]);


# #### Code 9.27

# In[25]:


with pm.Model() as m_9_5:
    a1 = pm.Normal("a1", 0, 10)
    a2 = pm.Normal("a2", 0, 10)

    mu = a1 + a2
    sigma = pm.Exponential("sigma", 1)

    yp = pm.Normal("y", mu, sigma, observed=y)

    m_9_5_trace = pm.sample(chains=3)

az.summary(m_9_5_trace, round_to=2)


# In[26]:


# Figure 9.11 (bottom half)
az.plot_trace(m_9_5_trace, figsize=[8, 4.5]);


# In[27]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv -w')


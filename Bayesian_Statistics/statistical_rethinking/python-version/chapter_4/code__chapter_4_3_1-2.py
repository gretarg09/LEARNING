
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
import arviz as az

from scipy.interpolate import griddata

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89  # sets default credible interval used by arviz


# #### Code 4.7 and 4.8
d = pd.read_csv("../Data/Howell1.csv", sep=";", header=0)
print(d.head())


 #### Code 4.9
print(az.summary(d.to_dict(orient="list"), kind="stats"))

# #### Code 4.10
print(d.height)


# #### Code 4.11
d2 = d[d.age >= 18]

# #### Code 4.12
plt.figure(figsize=(8, 4))
x = np.linspace(100, 250, 100)
plt.plot(x, stats.norm.pdf(x, 178, 20));
plt.savefig('f__4_12.png')


# #### Code 4.13
plt.figure(figsize=(8, 4))
x = np.linspace(-10, 60, 100)
plt.plot(x, stats.uniform.pdf(x, 0, 50));
plt.savefig('f__4_13.png')


# #### Code 4.14
# Prior predictive simulation
plt.figure(figsize=(8, 4))
n_samples = 1000
sample_mu = stats.norm.rvs(loc=178, scale=20, size=n_samples)
sample_sigma = stats.uniform.rvs(loc=0, scale=50, size=n_samples)
prior_h = stats.norm.rvs(loc=sample_mu, scale=sample_sigma)
az.plot_kde(prior_h)
plt.xlabel("heights")
plt.yticks([]);
plt.savefig('f__4_14.png')


# #### Code 4.15
# Prior predictive simulation
plt.figure(figsize=(8, 4))
n_samples = 1000
sample_mu = stats.norm.rvs(loc=178, scale=100, size=n_samples)
prior_h = stats.norm.rvs(loc=sample_mu, scale=sample_sigma)
az.plot_kde(prior_h)
plt.xlabel("heights")
plt.yticks([]);
plt.savefig('f__4_15.png')

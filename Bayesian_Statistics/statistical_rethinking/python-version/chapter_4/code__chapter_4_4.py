import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
import arviz as az
from scipy.interpolate import griddata

# #### Code 4.26
# We are repeating code 4.7, 4.8 and 4.10
d = pd.read_csv("../Data/Howell1.csv", sep=";", header=0)
d2 = d[d.age >= 18]


# #### Code 4.37
plt.plot(d2.height, d2.weight, ".")
plt.xlabel("height")
plt.ylabel("weight");
plt.savefig('f__4_37__height_weight.png')


# #### Code 4.38
# SIMULATE PRIOR PREDICTIVE DISTRIBUTION

height_rng = np.random.default_rng(2971)

N = 100  # 100 lines
a = stats.norm.rvs(178, 20, N)
b = stats.norm.rvs(0, 10, N)


# #### Code 4.39 and 4.40
# now we have 100 pairs of alpha and beta values. Now to plot the lines:
_, ax = plt.subplots(1, 2, sharey=True)
xbar = d2.weight.mean()
x = np.linspace(d2.weight.min(), d2.weight.max(), N)
for i in range(N):
    ax[0].plot(a[i] + b[i] * (x - xbar), "k", alpha=0.2)
    ax[0].set_xlim(d2.weight.min(), d2.weight.max())
    ax[0].set_ylim(-100, 400)
    ax[0].axhline(0, c="k", ls="--")
    ax[0].axhline(272, c="k")
    ax[0].set_xlabel("weight")
    ax[0].set_ylabel("height")
'''
 By looking at the simulated prior predictive distribution it is clear that the 
 Normal(0, 10) prior on beta is not a good choice. We can do better immediately. We know
 that average height increases with average weight, at least up to a point. Let's try to 
 restrict it a positive values. Teh easiest way to do this is to define the prior as
 Log-Normal instead. 

 Defining b as log-normal(0,1) means to claim that the logarithm of b has a normal(0,1) distribution.
 '''

b = stats.lognorm.rvs(s=1, scale=1, size=100)
for i in range(N):
    ax[1].plot(a[i] + b[i] * (x - xbar), "k", alpha=0.2)
    ax[1].set_xlim(d2.weight.min(), d2.weight.max())
    ax[1].set_ylim(-100, 400)
    ax[1].axhline(0, c="k", ls="--", label="embryo")
    ax[1].axhline(272, c="k")
    ax[1].set_xlabel("weight")
    ax[1].text(x=35, y=282, s="World's tallest person (272cm)")
    ax[1].text(x=35, y=-25, s="Embryo");

plt.savefig('f__4_40__prior_predictive_distribution.png')

# #### Code 4.42
# Fitting the linear model
with pm.Model() as m4_3:
    a = pm.Normal("a", mu=178, sigma=20)
    b = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    mu = a + b * (d2.weight.values - xbar)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=d2.height.values)
    trace_4_3 = pm.sample(1000, tune=1000)

with pm.Model() as m4_3b:
    a = pm.Normal("a", mu=178, sigma=20)
    b = pm.Normal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    mu = a + np.exp(b) * (d2.weight.values - xbar)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=d2.height.values)
    trace_4_3b = pm.sample(1000, tune=1000)


# #### Code 4.44
# Look at the marginal posterior distributions of the parameters.
print('Code 4_44 - marginal posterior distributions of the parameters')
print(az.summary(trace_4_3, kind="stats"))


# #### Code 4.45
# Look at the variance covariance matrix of the parameters.
trace_4_3_df = trace_4_3.posterior.to_dataframe()
print('Code 4.45 - variance covariance matrix of the parameters')
print(trace_4_3_df.cov().round(3))


# #### Code 4.46
# The code below plots the raw data, computes the posterior mean values for a and b, then draws the implied line:
plt.figure(figsize=(10, 5))
plt.plot(d2.weight, d2.height, ".")
plt.plot(
    d2.weight,
    trace_4_3.posterior["a"].mean().item(0)
    + trace_4_3.posterior["b"].mean().item(0) * (d2.weight - xbar),
)
plt.xlabel(d2.columns[1])
plt.ylabel(d2.columns[0]);
plt.savefig('f__4_46__raw_data_and_mean_posterior_line.png')

# #### Code 4.47
trace_4_3_df.head(5)

# #### Code 4.48
n = [10, 50, 150, 352][0]
dN = d2[:N]
# Fitting the linear model on the first 10 data points
with pm.Model() as m_N:
    a = pm.Normal("a", mu=178, sigma=100)
    b = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    mu = pm.Deterministic("mu", a + b * (dN.weight.values - dN.weight.mean()))
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=dN.height.values)
    chain_N = pm.sample(1000, tune=1000)

trace_N = az.extract_dataset(chain_N)

# #### Code 4.49
plt.plot(dN.weight, dN.height, "C0o")
nb_samples = trace_N.sizes["sample"]
idxs = height_rng.integers(nb_samples, size=20)
for idx in idxs:
    plt.plot(
        dN.weight,
        trace_N["a"].item(idx) + trace_N["b"].item(idx) * (dN.weight - dN.weight.mean()),
        "C1-",
        alpha=0.5,
    )
plt.xlabel(d2.columns[1])
plt.ylabel(d2.columns[0]);
plt.savefig('f__4_49__raw_data_and_20_posterior_lines.png')
plt.close()

# Alternative we can directly use the deterministic `mu` variable
#plt.plot(dN.weight, dN.height, "C0o")
#for idx in idxs:
#    plt.plot(d2.weight[:N], trace_N["mu"][:, idx], "C1-", alpha=0.5)
#plt.xlabel(d2.columns[1])
#plt.ylabel(d2.columns[0]);



N = [10, 50, 150, 352]

data = []
traces = []
for n in N:
    dN = d2[:n]
    with pm.Model() as m_N:
        a = pm.Normal("a", mu=178, sigma=100)
        b = pm.Lognormal("b", mu=0, sigma=1)
        sigma = pm.Uniform("sigma", 0, 50)
        mu = pm.Deterministic("mu", a + b * (dN.weight.values - dN.weight.mean()))
        height = pm.Normal("height", mu=mu, sigma=sigma, observed=dN.height.values)
        traces.append(pm.sample(1000, tune=1000, progressbar=False))
        data.append(dN)


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
cords = [(0, 0), (0, 1), (1, 0), (1, 1)]
for i in range(len(data)):
    idxs = height_rng.integers(nb_samples, size=N[i])
    ax[cords[i]].plot(data[i].weight, data[i].height, "C0o")
    for idx in idxs:
        ax[cords[i]].plot(
            data[i].weight, az.extract_dataset(traces[i])["mu"][:, idx], "C1-", alpha=0.5
        )
plt.savefig('f__4_50__raw_data_and_posterior_lines_for_10_50_150_352.png')


# #### Code 4.50
# A list of 10000 values of mu for an individual who weights 50 kg, by using sampling from posterior distribution.
data_4_3 = az.extract_dataset(trace_4_3)
mu_at_50 = data_4_3["a"] + data_4_3["b"] * (50 - d2.weight.mean())
# mu_at_50 is a vector of predicted means, one for each random sample from the posterior.

# #### Code 4.51
plt.figure(figsize=(10, 5))
az.plot_kde(mu_at_50.values)
plt.xlabel("heights")
plt.yticks([]);
plt.savefig('f__4_51__kde_heights_50_kg.png')
plt.close()

# #### Code 4.52
az.hdi(mu_at_50.values, hdi_prod=0.89)


# #### Code 4.53 and 4.54
'''
    We are doing _manually_, in the book is done using the ```link``` function.
    In the book on code 4.58 the following operations are performed _manually_.
'''

weight_seq = np.arange(25, 71)
# Given that we have a lot of samples we can use less of them for plotting (or we can use all!)
# GaG notes: We are basically just sampling 400 lines from the posterior distribution.
#            Then we calcuate a + b * (w - d2.weight.mean()) for each of the 400 lines.
#            That will give us samples of the mean for each weight value.
#            We can then use that to plot up the confidence interval.
nb_samples = trace_N.sizes["sample"]
trace_4_3_thinned = data_4_3.isel(sample=range(0, nb_samples, 10))

nb_samples_thinned = trace_4_3_thinned.sizes["sample"]

mu_pred = np.zeros((len(weight_seq), nb_samples_thinned))
for i, w in enumerate(weight_seq):
    mu_pred[i] = trace_4_3_thinned["a"] + trace_4_3_thinned["b"] * (w - d2.weight.mean())


# #### Code 4.55
plt.figure(figsize=(10, 5))
plt.plot(weight_seq, mu_pred, "C0.", alpha=0.1)
plt.xlabel("weight")
plt.ylabel("height")
plt.savefig('f__4_55__weight_sequence.png')
plt.close()


# #### Code 4.56
mu_mean = mu_pred.mean(1)
mu_hdi = az.hdi(mu_pred.T)


# #### Code 4.57
plt.figure(figsize=(10, 5))
az.plot_hdi(weight_seq, mu_pred.T)
plt.scatter(d2.weight, d2.height)
plt.plot(weight_seq, mu_mean, "k")
plt.xlabel("weight")
plt.ylabel("height")
plt.xlim(d2.weight.min(), d2.weight.max())
plt.savefig('f__4_57__weight_sequence.png')
plt.close()


# #### Code 4.59
# POSTERIOR PREDICTIVE DISTRIBUTION
'''
 Now we are going to use ```sample_posterior_predictive()``` from PyCM. This function gives us posterior predictive samples,
 that is for each value of the input variable we get a sample (from the posterior) of the output variable.
 Thus in the following example the shape of `height_pred['height'].shape is (200, 352)`
'''

samp_size = 100
slice_rate = int(len(trace_4_3["posterior"]["draw"]) / samp_size)
thin_data = trace_4_3.sel(draw=slice(None, None, slice_rate))
with m4_3:
    height_pred = pm.sample_posterior_predictive(thin_data)


# #### Code 4.60
height_pred_hdi = az.hdi(height_pred.posterior_predictive["height"], hdi_prob=0.89)


# #### Code 4.61
plt.figure(figsize=(10, 5))
ax = az.plot_hdi(weight_seq, mu_pred.T, hdi_prob=0.89)
az.plot_hdi(d2.weight, height_pred.posterior_predictive["height"], ax=ax, hdi_prob=0.89)
plt.scatter(d2.weight, d2.height)
plt.plot(weight_seq, mu_mean, "k")
plt.xlabel("weight")
plt.ylabel("height")
plt.xlim(d2.weight.min(), d2.weight.max());
plt.savefig('f__4_61.png')
plt.close()


# #### Code 4.62
# Change the number of samples used in 4.59 (200) to other values.
samp_size = 1000
slice_rate = int(len(trace_4_3["posterior"]["draw"]) / samp_size)
thin_data = trace_4_3.sel(draw=slice(None, None, slice_rate))
with m4_3:
    height_pred = pm.sample_posterior_predictive(thin_data)

plt.figure(figsize=(10, 5))
ax = az.plot_hdi(weight_seq, mu_pred.T)
az.plot_hdi(d2.weight, height_pred.posterior_predictive["height"], ax=ax, hdi_prob=0.89)
plt.scatter(d2.weight, d2.height)
plt.plot(weight_seq, mu_mean, "k")
plt.xlabel("weight")
plt.ylabel("height")
plt.xlim(d2.weight.min(), d2.weight.max());
plt.savefig('f__4_62.png')
plt.close()


# #### Code 4.63
# 
'''
Now we are going to generate heights from the posterior *manually*. Instead of restricting ourselves to the input values,
we are going to pass an array of equally spaced weights values, called `weight_seg`.

GaG notes: I think this is much easier to understand. It kind of explains how the sample_posterior_predictive() function works.
For each draw of the posterior, we are sampling a height value for each weight value in the weight_seq array.

In the code below we use the first 1000 samples from the posterior. We then calculate the mu for each of those samples.
Given the mu and the sigma we can draw a sample from a normal probability distribution. We do this for each weight 
value in the weight_seq array. This gives us 1000 samples for each weight value.  
'''

weight_seq = np.arange(25, 71)
post_samples = []
for _ in range(1000):  # number of samples from the posterior
    i = height_rng.integers(len(data_4_3))
    mu_pr = data_4_3["a"][i].item(0) + data_4_3["b"][i].item(0) * (weight_seq - d2.weight.mean())
    sigma_pred = data_4_3["sigma"][i]
    post_samples.append(height_rng.normal(mu_pr, sigma_pred))

plt.figure(figsize=(10, 5))
ax = az.plot_hdi(weight_seq, mu_pred.T, hdi_prob=0.89)
az.plot_hdi(weight_seq, np.array(post_samples), ax=ax, hdi_prob=0.89)
plt.scatter(d2.weight, d2.height)
plt.plot(weight_seq, mu_mean, "k")
plt.xlabel("weight")
plt.ylabel("height")
plt.xlim(d2.weight.min(), d2.weight.max());
plt.savefig('f__4_63.png')
plt.close()

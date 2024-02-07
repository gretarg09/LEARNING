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

# #### Code 4.27
with pm.Model() as m4_1:
    mu = pm.Normal("mu", mu=178, sigma=20)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=d2.height.values)


# #### Code 4.28
'''
 We could use a quadratic approximation like McElreath does in his book and we did in code 2.6 (chapter 2.4.4).
 But Using PyMC is really simple to just sample from the model using a "sampler method".
 Most common sampler methods are members of the Markov Chain Monte Carlo Method (MCMC) family 
 (for details read Section 2.4.3 and Chapter 8 of Statistical Rethinking).
 
 PyMC comes with various samplers. Some samplers are more suited than others for certain type of variable (and/or problems).
 For now we are going to let PyMC choose the sampler for us. PyMC also tries to provide a reasonable starting point for the simulation.
 By default PyMC uses the same adaptive procedure as in STAN `'jitter+adapt_diag'`, which starts with a identity mass matrix and 
 then adapts a diagonal based on the variance of the tuning samples. 
 
 You can read more details of PyMC [here](http://pymc-devs.github.io/pymc/notebooks/getting_started.html)
'''

with m4_1:
    trace_4_1 = pm.sample(1000, tune=1000)


az.plot_trace(trace_4_1)
plt.savefig('f__4_28__trace_plot.png')
# this function lets you check the samples values


# #### Code 4.29
print('m4_1 summary')
print(az.summary(trace_4_1, round_to=2, kind="stats"))


# #### Code 4.30
with pm.Model() as m4_1:
    mu = pm.Normal("mu", mu=178, sigma=20, testval=d2.height.mean())
    sigma = pm.Uniform("sigma", lower=0, upper=50, testval=d2.height.std())
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=d2.height.values)
    trace_4_1 = pm.sample(1000, tune=1000)


# #### Code 4.31
with pm.Model() as m4_2:
    mu = pm.Normal("mu", mu=178, sigma=0.1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=d2.height.values)
    trace_4_2 = pm.sample(1000, tune=1000)

print('m4_2 summary')
print(az.summary(trace_4_2, round_to=2, kind="stats"))


# #### Code 4.32
# For some computations could be nice to have the trace turned into a DataFrame,
# this can be done using the `trace_to_dataframe` function
trace_df = az.extract_dataset(trace_4_1).to_dataframe()
trace_df.cov()


# #### Code 4.33
np.diag(trace_df.cov())
trace_df.corr()


# #### Code 4.34
# We did not use the quadratic approximation, instead we use a MCMC method to sample from the posterior.
# Thus, we already have samples. We can do something like

trace_df.head()

# Or directly from the trace (we are getting the first ten samples of _sigma_)
trace_4_1.posterior["sigma"][0][:10]

# #### Code 4.35
# In our case, this is the same we did in the code 4.27
print('\ncode 4.35')
print(az.summary(trace_4_1, round_to=2, kind="stats"))

# #### Code 4.36
stats.multivariate_normal.rvs(mean=trace_df.mean(), cov=trace_df.cov(), size=10)

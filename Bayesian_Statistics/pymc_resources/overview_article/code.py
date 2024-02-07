import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# GENERATING DATA
# ----------------
# Initialize random number generator
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].scatter(X1, Y, alpha=0.6)
axes[1].scatter(X2, Y, alpha=0.6)
axes[0].set_ylabel("Y")
axes[0].set_xlabel("X1")
axes[1].set_xlabel("X2");

plt.savefig('f__synthetic_data.png')


# MODEL SPECIFICATIONS
import pymc as pm

print(f"Running on PyMC v{pm.__version__}")

basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)


with basic_model:
    # draw 1000 posterior samples
    idata = pm.sample() # The sample function runs the step methods assigned (or passed) to it for the given number of iterations and returns an InferenceData.

'''
The various attributes of the InferenceData object can be queried in a similar way to a dict containing a map from variable names to numpy.arrays.
For example, we can retrieve the sampling trace from the alpha latent variable by using the variable name as an index to the idata.posterior attribute.
The first dimension of the returned array is the chain index, the second dimension is the sampling index, while the later dimensions match the shape
of the variable. We can see the first 5 values for the alpha variable in each chain as follows:
'''

idata.posterior["alpha"].sel(draw=slice(0, 4))

'''
PyMC’s plotting and diagnostics functionalities are now taken care of by a dedicated, platform-agnostic package named Arviz. 
A simple posterior plot can be created using plot_trace.
'''
az.plot_trace(idata, combined=True);
plt.savefig('f__posterior_distributions.png')

# In addition, the summary function provides a text-based output of common posterior statistics:
az.summary(idata, round_to=2)

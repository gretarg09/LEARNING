import arvis as az
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import os


np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.linspace(0, 1, size)
X2 = np.linspace(0, 0.2, size)

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma


import pymc3 as pm
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP


basic_model = Model()

with basic_model:
    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=5)
    beta = Normal('beta', mu=0, sd=5, shape=2)
    sigma = HalfNormal('sigma', sd=4)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Deterministic variable, to have pyMC3 store mu as a value in the trace use
    # mu = Deterministic('mu', alpha + beta[0]*X1 + beta[1]*X2)

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

pm.model_to_graphviz(basic_model)


from pymc3 import NUTS, sample
from scipy import optimize

with basic_model:
    # obtain starting values via MAP
    # NOTE: it is generally not recommended to specify the starting values based on MAP.
    start = find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler
    step = NUTS(scaling=start)

    # draw 2000 posterior samples
    trace = sample(2000, step, start=start)



print(trace['alpha'])


from pymc3 import traceplot

traceplot(trace)

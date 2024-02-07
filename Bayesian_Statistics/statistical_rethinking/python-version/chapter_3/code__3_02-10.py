# import arviz as az
# import matplotlib.pyplot as plt
# import numpy as np
# import pymc as pm
# import scipy.stats as stats

# def uniform_prior(grid_points):
#     """ Returns Uniform prior density

#     grid_points (numpy.array): Array of prior values

#     Returns: density (numpy.array): Uniform density of prior values
#     """
#     return np.repeat(5, grid_points)


# def truncated_prior(grid_points, trunc_point=0.5):
#     """ Returns Truncated prior density

#     grid_points (numpy.array): Array of prior values
#     trunc_point (double): Value where the prior is truncated

#     Returns: density (numpy.array): Truncated density of prior values
#     """
#     return (np.linspace(0, 1, grid_points) >= trunc_point).astype(int)


# def double_exp_prior(grid_points):
#     """ Returns Double Exponential prior density

#     grid_points (numpy.array): Array of prior values

#     Returns: density (numpy.array): Double Exponential density of prior values
#     """
#     return np.exp(-5 * abs(np.linspace(0, 1, grid_points) - 0.5))



# def binom_post_grid_approx(prior_func, grid_points=5, success=6, tosses=9):
#     """ Returns the grid approximation of posterior distribution with binomial likelihood.

#     prior_func (function): A function that returns the likelihood of the prior
#     grid_points (int): Number of points in the prior grid
#     successes (int): Number of successes
#     tosses (int): number of tosses

#     Returns: p_grid (numpy.array): Array of prior values
#              posterior (numpy.array): Likelihood (density) of prior values
#     """
#     # define grid
#     p_grid = np.linspace(0, 1, grid_points)

#     # define prior
#     prior = prior_func(grid_points)

#     # compute likelihood at each point in the grid
#     likelihood = stats.binom.pmf(success, tosses, p_grid)

#     # compute product of likelihood and prior
#     unstd_posterior = likelihood * prior

#     # standardize the posterior, so it sums to 1
#     posterior = unstd_posterior / unstd_posterior.sum()
#     return p_grid, posterior


# p_grid, posterior = binom_post_grid_approx(uniform_prior, grid_points=100, success=6, tosses=9)
# samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)


# _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
# ax0.plot(samples, "o", alpha=0.2)
# ax0.set_xlabel("sample number")
# ax0.set_ylabel("proportion water (p)")
# az.plot_kde(samples, ax=ax1)
# ax1.set_xlabel("proportion water (p)")
# ax1.set_ylabel("density");


# MY VERSION

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as stats
from models import globe_tossing_model


'''Now we wish to draw 10000 samples from the posterior distribution. 
Within the bucket, each value exist in proportion to its posterior probability, 
such that values near the peak are much more common than those in the tails.
We're going to scoop out 10000 values fro mthe bucket. Provided that the bucket
is well mixed, the resulting samples will have the same proportions as the 
exact posterior distribution.
'''

p_grid, posterior = globe_tossing_model(1000)

samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)

_, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.plot(samples, "o", alpha=0.2)
ax0.set_xlabel("sample number")
ax0.set_ylabel("proportion water (p)")
az.plot_kde(samples, ax=ax1)
ax1.set_xlabel("proportion water (p)")
ax1.set_ylabel("density");

plt.show()

# 3.6
# Add up posterior probability where p > 0.5
print('\n3.6')
print(sum(posterior[p_grid < 0.5]))


# 3.7
# Let's now see how to perform the same calculation, using samples from the posterior distribution
# rather than the exact posterior distribution. This approach does generalze to complex models with many parameters
# and so you can use it anywhere
print('\n3.7')
print(sum(samples < 0.5) / 1e4)


# 3.8
# how much posterior probability lies between p = 0.5 and p = 0.75?
print('\n3.8')
print(sum((samples > 0.5) & (samples < 0.75)) / 1e4)

#. 3.9
# lets calculate the confidence interval or in other words the compatibility interval
# What the interval indicates is a range of parameter values compatible with the model and the data.
print('\n3.9')
print(np.percentile(samples, 80))

# 3.10
print('\n3.10')
print(np.percentile(samples, [10, 90]))





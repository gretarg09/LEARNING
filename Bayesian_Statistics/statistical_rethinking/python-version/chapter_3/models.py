import numpy as np
from scipy import stats


'''Before beginning to work with samples, we need to generate them. Here's a reminderfor how to compute 
the posterior for the globe tossing model, using grid approximation. 
Remember, the posterior here means the probability of p conditional on the data.
'''

def globe_tossing_model(number_of_points, number_of_success=6, number_of_trials=9): 

    # number of points on the grid
    n = number_of_points

   # define the grid
    p_grid = np.linspace(start=0, stop=1, num=n)

    # define the prior
    prior = np.repeat(1, n) # uniform

    # k = number of successes
    # n = number of trials
    # p = probability of success

    # compute likelihood at each value in grid
    # k = number of successes
    # n = number of trials
    # p = probability of success
    likelihood = stats.binom.pmf(k=number_of_success, n=number_of_trials, p=p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()

    return p_grid, posterior

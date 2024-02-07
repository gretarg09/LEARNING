import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as stats


'''
Notes:

    * Remember, the goal is to count all the ways the data could arise, given the assumptions. This means,
    as in the globe tossing model, that for each possible value of the unobserved variables, such as p, 
    we need to define the relative number of ways -the probablitites- that the values fo each observed
    variable could arise. An then for each unobserved variable, we need to define the prior plausability of 
    each value it could take.
     
    * For the count of water W and land L, we define how plausible any combination of W and L would be, 
    for a specific value of p. This is very much like the marble counting we did earlier in the chapter.
    Each specific value of p corresponds to a specific plausibility of the data.

    * The counts of "water" W and "land" L are distbuted binomially, with probability p of "water" on each
    toss.
'''

'''
    In the context of the globe tossing problem, grid approcimation works extremely well. 
    So let's build a grid approximation for the model we've constructred so far. Here is the recipy:


    1. Define the grid. This means you decide how many points to use in estimating the posterior, 
    and then you make a list of the parameter values on the grid.
    2. Compute the value of the prior at each parameter value on the grid.
    3. Compute the likelyhood at each parameter value.
    4. Compute the unstandardized posterior at each parameter value, by multiplying the prior by the
    likelyhood.
    5. Finally, standardize the posterior, by dividing each value by the sum of all values. 

    In the globe tossing context, here's the code to complete all five of these steps:
'''


# number of points on the grid
n = 100

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
likelihood = stats.binom.pmf(k=6, n=9, p=p_grid)

# compute product of likelihood and prior
unstd_posterior = likelihood * prior

# standardize the posterior, so it sums to 1
posterior = unstd_posterior / unstd_posterior.sum()


_, ax = plt.subplots(1, 1)

ax.plot(p_grid, posterior, "o-")
ax.set_xlabel("probability of water")
ax.set_ylabel("posterior probability")

plt.show()

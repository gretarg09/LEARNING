import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(0)


mu = 170
sd = 7


# generate samples from our distribution, draw from a normal distribution.
x = norm.rvs(loc=mu, scale=sd, size=100)

# maximum likelihood mean
print('\n the mean')
print(x.mean())

# maximum likelihood variance 
print('\n the variance')
print(x.var())

# maximum likelihood std
print('\n the standard deviation')
print(x.std())


# unbiased variance
print('\n the unbiased variance')
print(x.var(ddof=1))

# unbiased std
print('\n the unbiased standard deviation')
print(x.std(ddof=1))

# at what height are you in the 95th percentile?
print('\n the 95th percentile')
print(norm.ppf(0.95, loc=mu, scale=sd))

# you are 160 cm tall, what percentile are you in?
print('\n the percentile of 160 cm')
print(norm.cdf(160, loc=mu, scale=sd))

# you are 180 cm tall, what is the probability that someone is taller than you?
print('\n the probability that someone is taller than you')
print(1 - norm.cdf(183, loc=mu, scale=sd))

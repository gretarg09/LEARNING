import pymc3 as pm
import arviz as az


'''
    https://dr-robert-kuebler.medium.com/a-gentle-introduction-to-bayesian-inference-6a7552e313cb
    https://towardsdatascience.com/conducting-bayesian-inference-in-python-using-pymc3-d407f8d934a5 
'''
tosses = [
    1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
    0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
    0, 1, 1, 0, 1, 1, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
    1, 1, 1, 0, 0, 0, 1, 0, 1, 0
]

with pm.Model() as model:
    # Define the prior
    # PyMC3 distributions always want a name, that’s always the first parameter you have to specify.
    # Usually, I just use the variable name again. 
    theta = pm.Beta('theta', 2, 2)
    
    # define the likelihood
    data = pm.Bernoulli('data', theta, observed=tosses)
    
    # get the samples
    # We obtain some trace object, but just imagine that it contains a lot of numbers
    # from the posterior distribution. Wanna peek inside? Use trace.get_values('theta').
    # You will receive a numpy array containing the samples you can then play around with.
    trace = pm.sample(return_inferencedata=True)

    az.plot_posterior(trace, hdi_prob=0.99)

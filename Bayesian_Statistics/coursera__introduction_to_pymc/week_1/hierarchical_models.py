'''
Problem statement 

W measure the water samples for the three districts, and we collect 30 samples for each district. The data is
simply a binary value that indicates whether the water is contaminated or not. We count the number of samples that 
have contamination below the acceptable levels. We generate three arrays:

    * N_samples: The total number of samples collected in each district
    * G_samples: The number of good samples or samples with contamination levels below a certain threshold.
    * group_idx : The id for each district or group.
'''

import numpy as np
import pymc3 as pm
import arviz as az
from matplotlib import pyplot as plt


N_samples = [30, 30, 30] # total number of samples collected in each district
G_samples = [4, 15, 22] # Number of samples with water contamination below acceptable levels.

# Create an Id for each of the 30 + 30 + 30 samples - 0, 1, 2 to indicate that they belong to
# different groups.


group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []

for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i] - G_samples[i]]))


def get_hyperprior_model(data, N_samples, group_idx):
    with pm.Model() as model_h:
        m = pm.Beta('m', 1., 1.) # Hyperprior
        x = pm.HalfNormal('x', sd=10) # Hyperprior

        alpha = pm.Deterministic('alpha', m * x)
        beta = pm.Deterministic('beta', (1 - m) * x)

        theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=len(N_samples))
        y = pm.Bernoulli('y', p=theta[group_idx], observed=data)


        trace_h = pm.sample(2000)

    az.plot_trace(trace_h)
    plt.show()
    print(az.summary(trace_h))
    return model_h

model = get_hyperprior_model(data, N_samples, group_idx) 
pm.model_to_graphviz(model)

# Shrinkage in Hierarchical Models

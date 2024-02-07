import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as stats
from models import globe_tossing_model


p_grid, posterior = globe_tossing_model(number_of_points=1000, number_of_trials=3, number_of_success=3)
plt.plot(p_grid, posterior)
plt.xlabel("proportion water (p)")
plt.ylabel("Density");

# 3.12
print('\n3.12')
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)
print(np.percentile(samples, [25, 75]))

# 3.1#
print('\n3.13')
print(az.hdi(samples, hdi_prob=0.5))

plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as stats


# In[3]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use('arviz-darkgrid')


# #### Code 3.1
# 
# $$Pr(vampire|positive) = \frac{Pr(positive|vampire) Pr(vampire)} {Pr(positive)}$$
# 
# $$Pr(positive) = Pr(positive|vampire) Pr(vampire) + Pr(positive|mortal) 1 − Pr(vampire)$$

# In[4]:


PrPV = 0.95
PrPM = 0.01
PrV = 0.001
PrP = PrPV * PrV + PrPM * (1 - PrV)
PrVP = PrPV * PrV / PrP
PrVP


# #### Code 3.2 - 3.5
# 
# We are goint to use the same function we use on chapter 2 (code 2.3)

# In[5]:


def posterior_grid_approx(grid_points=100, success=6, tosses=9):
    """
    """
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(5, grid_points)  # uniform
    #prior = (p_grid >= 0.5).astype(int)  # truncated
    #prior = np.exp(- 5 * abs(p_grid - 0.5))  # double exp

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior


# In[6]:


p_grid, posterior = posterior_grid_approx(grid_points=100, success=6, tosses=9)
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)


# In[7]:


_, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
ax0.plot(samples, 'o', alpha=0.2)
ax0.set_xlabel('sample number', fontsize=14)
ax0.set_ylabel('proportion water (p)', fontsize=14)
az.plot_kde(samples, ax=ax1)
ax1.set_xlabel('proportion water (p)', fontsize=14)
ax1.set_ylabel('density', fontsize=14);


# #### Code 3.6

# In[8]:


sum(posterior[ p_grid < 0.5 ])


# #### Code 3.7

# In[9]:


sum( samples < 0.5 ) / 1e4


# #### Code 3.8

# In[10]:


sum((samples > 0.5) & (samples < 0.75)) / 1e4


# #### Figure 3.2
# 
# 

# In[11]:


# plotting out intervals of defined boundaries: 

# wider figure
plt.figure(figsize=(20,20)) 

### Intervals of defined boundaries:

# plot p < 0.5
plt.subplot(2, 2, 1)
plt.plot(p_grid, posterior)
plt.xlabel('proportion of water (p)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks([0,0.25,0.50,0.75,1.00])
plt.fill_between(p_grid, posterior, where = p_grid < 0.5)

# plot p < 0.5 & p > 0.75
plt.subplot(2, 2, 2)
plt.plot(p_grid, posterior)
plt.xlabel('proportion of water (p)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks([0,0.25,0.50,0.75,1.00])
plt.fill_between(p_grid, posterior, where = (p_grid > 0.5)&(p_grid < 0.75))

### Intervals of defined mass:

# plot p < 0.5
plt.subplot(2, 2, 3)
plt.plot(p_grid, posterior, label = "lower 80%")
plt.xlabel('proportion of water (p)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks([0,0.25,0.50,0.75,1.00])
plt.fill_between(p_grid, posterior, where = p_grid < np.percentile(samples, 80))
plt.legend(loc=0)

# plot p < 0.5 & p > 0.75
perc_range = np.percentile(samples, [10, 90])
plt.subplot(2, 2, 4)
plt.plot(p_grid, posterior,label = "middle 80%")
plt.xlabel('proportion of water (p)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks([0,0.25,0.50,0.75,1.00])
plt.fill_between(p_grid, posterior, where = (p_grid > perc_range[0])&(p_grid < perc_range[1]))
plt.legend(loc=0);


# #### Code 3.9

# In[12]:


np.percentile(samples, 80)


# #### Code 3.10

# In[13]:


np.percentile(samples, [10, 90])


# #### Code 3.11

# In[14]:


p_grid, posterior = posterior_grid_approx(success=3, tosses=3)
plt.plot(p_grid, posterior)
plt.xlabel('proportion water (p)', fontsize=14)
plt.ylabel('Density', fontsize=14);


# #### Code 3.12

# In[15]:


samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)
np.percentile(samples, [25, 75])


# #### Code 3.13

# In[16]:


az.hpd(samples, credible_interval=0.5)


# #### Figure 3.3 

# In[17]:


# wider figure
plt.figure(figsize=(20,6)) 

# calculate posterior: 
p_grid, posterior = posterior_grid_approx(success=3, tosses=3)

# PI
pi_interval = np.percentile(samples, [25, 75])

# PI Plot
plt.subplot(1, 2, 1)
plt.plot(p_grid, posterior)
plt.xlabel('proportion water (p)', fontsize=14)
plt.ylabel('density', fontsize=14)
plt.fill_between(p_grid, posterior, where = (p_grid > pi_interval[0]) & (p_grid < pi_interval[1]))
plt.title('50% Percentile Interval')

# HDPI
hdpi_interval = az.hpd(samples, credible_interval=0.5)

# HDPI Plot
plt.subplot(1, 2, 2)
plt.plot(p_grid, posterior)
plt.xlabel('proportion water (p)', fontsize=14)
plt.ylabel('density', fontsize=14)
plt.fill_between(p_grid, posterior, where = (p_grid > hdpi_interval[0]) & (p_grid < hdpi_interval[1]))
plt.title('50% HDPI');


# #### Code 3.14

# In[18]:


p_grid[posterior == max(posterior)]


# #### Code 3.15

# In[19]:


stats.mode(samples)[0]


# #### Code 3.16

# In[20]:


np.mean(samples), np.median(samples)


# #### Code 3.17

# In[21]:


sum(posterior * abs(0.5 - p_grid))


# #### Code 3.18 and 3.19

# In[22]:


loss = [sum(posterior * abs(p - p_grid)) for p in p_grid]
p_grid[loss == min(loss)]


# #### A portion of Figure 3.4

# In[23]:


plt.plot(p_grid, loss, markevery =p_grid[loss == min(loss)][0], marker = "o", label = "min loss")
plt.xlabel('decision', fontsize=14)
plt.ylabel('expected proportional loss', fontsize=14)
plt.title(f'Loss Function')
plt.legend(loc=0);


# #### Code 3.20

# In[24]:


stats.binom.pmf(range(3), n=2, p=0.7)


# #### Code 3.21

# In[25]:


stats.binom.rvs(n=2, p=0.7, size=1)


# #### Code 3.22

# In[26]:


stats.binom.rvs(n=2, p=0.7, size=10)


# #### Code 3.23

# In[27]:


dummy_w = stats.binom.rvs(n=2, p=0.7, size=int(1e5))
[(dummy_w == i).mean() for i in range(3)]


# #### Code 3.24, 3.25 and 3.26

# In[28]:


dummy_w = stats.binom.rvs(n=9, p=0.7, size=int(1e5))
#dummy_w = stats.binom.rvs(n=9, p=0.6, size=int(1e4))
#dummy_w = stats.binom.rvs(n=9, p=samples)
plt.hist(dummy_w, bins=50)
plt.xlabel('dummy water count', fontsize=14)
plt.ylabel('Frequency', fontsize=14);


# #### Code 3.27

# In[29]:


p_grid, posterior = posterior_grid_approx(grid_points=100, success=6, tosses=9)
np.random.seed(100)
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)


# #### Code 3.28

# In[30]:


birth1 = np.array([1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0, 0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0, 1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,1,1,1,1])
birth2 = np.array([0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,
1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,
0,0,0,1,1,1,0,0,0,0])


# #### Code 3.29

# #### Code 3.30

# In[31]:


sum(birth1) + sum(birth2)


# In[32]:


import platform
import sys

import IPython
import matplotlib
import scipy

print("""This notebook was created using:\nPython {}\nIPython {}\nPyMC {}\nArviZ {}\nNumPy {}\nSciPy {}\nMatplotlib {}\n""".format(sys.version[:5], IPython.__version__, pm.__version__, az.__version__, np.__version__, scipy.__version__, matplotlib.__version__))


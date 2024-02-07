#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy as sp
import seaborn as sns

from matplotlib.offsetbox import AnchoredText


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use('arviz-darkgrid')


# In[3]:


SEED = 1234567890
np.random.seed(SEED)


# #### Code 8.1

# In[4]:


num_weeks = int(1e5)
positions = np.empty(num_weeks, dtype=np.int64)
current = 9

for i in range(num_weeks):
    # record current position
    positions[i] = current

    # flip coin to generate proposal
    proposal = current + np.random.choice([-1, 1])
    # now make sure he loops around the archipelago
    proposal %= 10
    
    #move?
    prob_move = (proposal + 1) / (current + 1)
    current = proposal if np.random.uniform() < prob_move else current


# In[5]:


_, (week_ax, island_ax) = plt.subplots(ncols=2, figsize=(16, 6))

week_ax.scatter(np.arange(100) + 1, positions[:100] + 1);

week_ax.set_xlabel('week');
week_ax.set_ylim(0, 11);
week_ax.set_ylabel('island');

island_ax.bar(np.arange(10) + 0.6, np.bincount(positions));

island_ax.set_xlim(0.4, 10.6);
island_ax.set_xlabel('island');
island_ax.set_ylabel('number of weeks');


# #### Code 8.2

# In[6]:


rugged_df = (pd.read_csv('Data/rugged.csv', sep=';')
               .assign(log_gdp=lambda df: np.log(df.rgdppc_2000))
               .dropna(subset=['log_gdp']))


# #### Code 8.3

# In[7]:


with pm.Model() as m8_1_map:
    a = pm.Normal('a', 0., 100.)
    bR = pm.Normal('bR', 0., 10.)
    bA = pm.Normal('bA', 0., 10.)
    bAR = pm.Normal('bAR', 0., 10.)
    mu = a \
            + bR * rugged_df.rugged \
            + bA * rugged_df.cont_africa \
            + bAR * rugged_df.rugged * rugged_df.cont_africa
    
    sigma = pm.Uniform('sigma', 0., 10.)
    
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=rugged_df.log_gdp)


# In[8]:


with m8_1_map:
    map_8_1 = pm.find_MAP()


# In[9]:


map_8_1


# #### Code 8.5

# In[11]:


with pm.Model() as m8_1:
    a = pm.Normal('a', 0., 100.)
    bR = pm.Normal('bR', 0., 10.)
    bA = pm.Normal('bA', 0., 10.)
    bAR = pm.Normal('bAR', 0., 10.)
    mu = a \
            + bR * rugged_df.rugged \
            + bA * rugged_df.cont_africa \
            + bAR * rugged_df.rugged * rugged_df.cont_africa
    
    sigma = pm.HalfCauchy('sigma', 2.)
    
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=rugged_df.log_gdp)


# In[12]:


with  m8_1:
    trace_8_1 = pm.sample(1000, tune=1000)


# In[13]:


pm.summary(trace_8_1, alpha=.11).round(2)


# #### Code 8.7

# In[14]:


with m8_1:
    trace_8_1_4_chains = pm.sample(1000, tune=1000)


# In[18]:


az.summary(trace_8_1_4_chains, credible_interval=.89).round(2)


# #### Code 8.8

# In[19]:


trace_8_1_df = pm.trace_to_dataframe(trace_8_1)


# In[20]:


trace_8_1_df.head()


# #### Code 8.9 and 8.10

# In[21]:


def plot_corr(x, y, **kwargs):
    corrcoeff = np.corrcoef(x, y)[0, 1]
    
    artist = AnchoredText(f'{corrcoeff:.2f}', loc=10)
    plt.gca().add_artist(artist)
    plt.grid(b=False)

trace_8_1_df = pm.trace_to_dataframe(trace_8_1_4_chains)
grid = (sns.PairGrid(trace_8_1_df,
                     x_vars=['a', 'bR', 'bA', 'bAR', 'sigma'],
                     y_vars=['a', 'bR', 'bA', 'bAR', 'sigma'],
                     diag_sharey=False)
           .map_diag(sns.kdeplot)
           .map_upper(plt.scatter, alpha=0.1)
           .map_lower(plot_corr))


# #### Code 8.11

# In[22]:


m8_1.logp({
    varname: trace_8_1[varname].mean()
        for varname in trace_8_1.varnames})


# The computation of DIC has been deprecated and is no longer available in PyMC

# In[25]:


az.waic(trace_8_1)


# #### Code 8.12

# In[26]:


az.plot_trace(trace_8_1);


# #### Code 8.13

# In[27]:


y = np.array([-1., 1.])

with pm.Model() as m8_2:
    alpha = pm.Flat('alpha')
    sigma = pm.Bound(pm.Flat, lower=0.)('sigma')
    
    y_obs = pm.Normal('y_obs', alpha, sigma, observed=y)


# In[28]:


with m8_2:
    trace_8_2 = pm.sample(draws=2000, tune=2000)


# In[29]:


az.plot_trace(trace_8_2);


# In[34]:


az.effective_sample_size(trace_8_2)


# #### Code 8.14

# In[35]:


az.summary(trace_8_2, credible_interval=.89).round(2)


# #### Code 8.15

# In[36]:


with pm.Model() as m8_3:
    alpha = pm.Normal('alpha', 1., 10.)
    sigma = pm.HalfCauchy('sigma', 1.)
    
    y_obs = pm.Normal('y_obs', alpha, sigma, observed=y)


# In[37]:


with m8_3:
    trace_8_3 = pm.sample(1000, tune=1000)


# In[38]:


az.summary(trace_8_3, credible_interval=.89).round(2)


# In[39]:


az.plot_trace(trace_8_3);


# #### Code 8.16

# In[40]:


y = sp.stats.cauchy.rvs(0., 5., size=int(1e4))
mu = y.cumsum() / (1 + np.arange(int(1e4)))


# In[41]:


plt.plot(mu);


# #### Code 8.17

# In[42]:


y = np.random.normal(0., 1., size=100)


# #### Code 8.18

# In[43]:


with pm.Model() as m8_4:
    a1 = pm.Flat('a1')
    a2 = pm.Flat('a2')
    sigma = pm.HalfCauchy('sigma', 1.)
    
    y_obs = pm.Normal('y_obs', a1 + a2, sigma, observed=y)


# In[44]:


with m8_4:
    trace_8_4 = pm.sample(1000, tune=1000)


# In[47]:


az.summary(trace_8_4, credible_interval=.89).round(2)


# In[48]:


az.plot_trace(trace_8_4);


# #### Code 8.19

# In[49]:


with pm.Model() as m8_5:
    a1 = pm.Normal('a1', 0., 10.)
    a2 = pm.Normal('a2', 0., 10.)
    sigma = pm.HalfCauchy('sigma', 1.)
    
    y_obs = pm.Normal('y_obs', a1 + a2, sigma, observed=y)


# In[50]:


with m8_5:
    trace_8_5 = pm.sample(1000, tune=1000)


# In[51]:


az.summary(trace_8_5, credible_interval=.89).round(2)


# In[52]:


az.plot_trace(trace_8_5);


# In[53]:


import platform
import sys

import IPython
import matplotlib
import scipy

print("""This notebook was created using:\nPython {}\nIPython {}\nPyMC {}\nArviZ {}\nNumPy {}\nSciPy {}\nMatplotlib {}\n""".format(sys.version[:5], IPython.__version__, pm.__version__, az.__version__, np.__version__, scipy.__version__, matplotlib.__version__))


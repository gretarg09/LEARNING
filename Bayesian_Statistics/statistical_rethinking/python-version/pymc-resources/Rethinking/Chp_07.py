#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.formula.api as smf

from scipy import stats
from scipy.interpolate import griddata


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use('arviz-darkgrid')


# #### Code 7.1

# In[3]:


d = pd.read_csv('Data/rugged.csv', sep=';', header=0)
#d.head()

# make log version of outcome
d['log_gdp'] = np.log(d.rgdppc_2000)

# extract countries with GDP data
dd = d[np.isfinite(d['rgdppc_2000'])]

# split countries into Africa and non-Africa
dA1 = dd[dd.cont_africa==1]  # Africa
dA0 = dd[dd.cont_africa==0]  # not Africa


# #### Code 7.2

# In[4]:


# Fit the regression models with this code.
# African nations
with pm.Model() as model_7_2:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    # good (default) alternatives for sigma (in this and other models) are
    # sigma = pm.HalfNormal('sigma', 5)
    # sigma = pm.HalfCauchy('sigma', 5)
    # some people recomed avoiding "hard" boundaries unless they have a theoretical/data-based justification, like a correlation that is restricted to be [-1, 1].
    mu = pm.Deterministic('mu', a + bR * dA1['rugged'])
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dA1['rgdppc_2000']))
    trace_7_2 = pm.sample(1000, tune=1000)


# In[5]:


varnames = ['~mu']
pm.traceplot(trace_7_2, varnames);


# In[6]:


# non-African nations
with pm.Model() as model_7_2_2:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    # good (default) alternatives for sigma (in this and other models) are
    # sigma = pm.HalfNormal('sigma', 5)
    # sigma = pm.HalfCauchy('sigma', 5)
    # some people recomed avoiding "hard" boundaries unless they have a theoretical/data-based justification, like a correlation that is restricted to be [-1, 1].
    mu = pm.Deterministic('mu', a + bR * dA0['rugged'])
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dA0['rgdppc_2000']))
    trace_7_2_2 = pm.sample(1000, tune=1000)


# In[7]:


pm.traceplot(trace_7_2_2, varnames);


# In[8]:


# Plot the data

mu_mean = trace_7_2['mu']
mu_hpd = pm.hpd(mu_mean)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,3))
ax1.plot(dA1['rugged'], np.log(dA1['rgdppc_2000']), 'C0o')
ax1.plot(dA1['rugged'], mu_mean.mean(0), 'C1')
az.plot_hpd(dA1['rugged'], mu_mean, ax=ax1)
ax1.set_title('Africa')
ax1.set_ylabel('log(rgdppc_2000)');
ax1.set_xlabel('rugged')

mu_mean = trace_7_2_2['mu']

ax2.plot(dA0['rugged'], np.log(dA0['rgdppc_2000']), 'ko')
ax2.plot(dA0['rugged'], mu_mean.mean(0), 'C1')
ax2.set_title('not Africa')
ax2.set_ylabel('log(rgdppc_2000')
ax2.set_xlabel('rugged')
az.plot_hpd(dA0['rugged'], mu_mean, ax=ax2);


# #### Code 7.3

# In[9]:


# Model the entire data
with pm.Model() as model_7_3:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bR * dd.rugged)
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_3 = pm.sample(1000, tune=1000)


# #### Code 7.4

# In[10]:


# Model the entire data including a dummy variable
with pm.Model() as model_7_4:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    bA = pm.Normal('bA', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bR * dd.rugged + bA * dd.cont_africa)
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_4 = pm.sample(1000, tune=1000)


# #### Code 7.5
# 
# WAIC values are point estimates and hence is a good idea to include the uncertainty asociated with their estimation when computing weights. PyMC uses a Bayesian bootstrapping to do this (read more [here](https://arxiv.org/abs/1704.02030)), and also to compute the standard error (SE) of WAIC/LOO estimates. If you set `bootstrapping = False` weights (and SE) will be computed as in the book.

# In[11]:


comp_df = az.compare({'m7.3' : trace_7_3, 
                      'm7.4' : trace_7_4})
comp_df


# In[12]:


az.plot_compare(comp_df);


# #### Code 7.6
# 
# Since the link function isn't implemented we have to compute the mean over samples ourselves using a loop.

# In[13]:


rugged_seq = np.arange(-1, 9, 0.25)

# compute mu over samples
mu_pred_NotAfrica = np.zeros((len(rugged_seq), len(trace_7_4['bR'])))
mu_pred_Africa = np.zeros((len(rugged_seq), len(trace_7_4['bR'])))

for iSeq, seq in enumerate(rugged_seq):
    mu_pred_NotAfrica[iSeq] = trace_7_4['a'] + trace_7_4['bR'] * rugged_seq[iSeq] + trace_7_4['bA'] * 0
    mu_pred_Africa[iSeq] = trace_7_4['a'] + trace_7_4['bR'] * rugged_seq[iSeq] + trace_7_4['bA'] * 1 


# In[14]:


# summarize to means and intervals
mu_mean_NotAfrica = mu_pred_NotAfrica.mean(1)
mu_mean_Africa = mu_pred_Africa.mean(1)


# In[15]:


plt.plot(dA1['rugged'], np.log(dA1['rgdppc_2000']), 'C0o')
plt.plot(rugged_seq, mu_mean_Africa, 'C0')
az.plot_hpd(rugged_seq, mu_pred_Africa.T, credible_interval=0.97, color='C0')
plt.plot(dA0['rugged'], np.log(dA0['rgdppc_2000']), 'ko')
plt.plot(rugged_seq, mu_mean_NotAfrica, 'k')
az.plot_hpd(rugged_seq, mu_pred_NotAfrica.T, credible_interval=0.97, color='k')
plt.annotate('not Africa', xy=(6, 9.5))
plt.annotate('Africa', xy=(6, 6))
plt.ylabel('log(rgdppc_2000)')
plt.xlabel('rugged');


# #### Code 7.7

# In[16]:


with pm.Model() as model_7_5:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    bA = pm.Normal('bA', mu=0, sd=1)
    bAR = pm.Normal('bAR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    gamma = bR + bAR * dd.cont_africa
    mu = pm.Deterministic('mu', a + gamma * dd.rugged + bA * dd.cont_africa)
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_5 = pm.sample(1000, tune=1000)


# #### Code 7.8

# In[17]:


comp_df = az.compare({'m7.3': trace_7_3,
                      'm7.4' : trace_7_4,
                      'm7.5' : trace_7_5})

comp_df


# In[18]:


az.plot_compare(comp_df);


# #### Code 7.9

# In[19]:


with pm.Model() as model_7_5b:
    a = pm.Normal('a', mu=8, sd=100)
    bR = pm.Normal('bR', mu=0, sd=1)
    bA = pm.Normal('bA', mu=0, sd=1)
    bAR = pm.Normal('bAR', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bR*dd.rugged + bAR*dd.rugged*dd.cont_africa + bA*dd.cont_africa)
    log_gdp = pm.Normal('log_gdp', mu, sigma, observed=np.log(dd.rgdppc_2000))
    trace_7_5b = pm.sample(1000, tune=1000)


# #### Code 7.10
# First calculate the necessary posterior predicted means. The link function is replaced by a loop. We'll use model 7.5b since it's a one-liner.
# 

# In[20]:


rugged_seq = np.arange(-1, 9, 0.25)

# compute mu over samples
mu_pred_NotAfrica = np.zeros((len(rugged_seq), len(trace_7_5b['bR'])))
mu_pred_Africa = np.zeros((len(rugged_seq), len(trace_7_5b['bR'])))
for iSeq, seq in enumerate(rugged_seq):
    mu_pred_NotAfrica[iSeq] = trace_7_5b['a'] + trace_7_5b['bR']*rugged_seq[iSeq] + \
                              trace_7_5b['bAR']*rugged_seq[iSeq]*0 +\
                              trace_7_5b['bA'] * 0
    mu_pred_Africa[iSeq] = trace_7_5b['a'] + trace_7_5b['bR']*rugged_seq[iSeq] + \
                              trace_7_5b['bAR']*rugged_seq[iSeq]*1 +\
                              trace_7_5b['bA'] * 1


# In[21]:


# summarize to means and intervals
mu_mean_NotAfrica = mu_pred_NotAfrica.mean(1)
mu_mean_Africa = mu_pred_Africa.mean(1)


# #### Code 7.11

# In[22]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,3))
ax1.plot(dA1['rugged'], np.log(dA1['rgdppc_2000']), 'C0o')
ax1.plot(rugged_seq, mu_mean_Africa, 'C0')
az.plot_hpd(rugged_seq, mu_pred_Africa.T, credible_interval=0.97, color='C0', ax=ax1)

ax1.set_title('African Nations')
ax1.set_ylabel('log GDP year 2000', fontsize=14);
ax1.set_xlabel('Terrain Ruggedness Index', fontsize=14)
               
ax2.plot(dA0['rugged'], np.log(dA0['rgdppc_2000']), 'ko')
ax2.plot(rugged_seq, mu_mean_NotAfrica, 'k')
az.plot_hpd(rugged_seq, mu_pred_NotAfrica.T, credible_interval=0.97, color='C1', ax=ax2)
ax2.set_title('Non-African Nations')
ax2.set_ylabel('log GDP year 2000', fontsize=14)
ax2.set_xlabel('Terrain Ruggedness Index', fontsize=14);


# #### Code 7.12

# In[23]:


varnames = ['~mu']
az.summary(trace_7_5b, varnames, credible_interval=.89).round(3)


# #### Code 7.13

# In[24]:


gamma_Africa = trace_7_5b['bR'] + trace_7_5b['bAR'] * 1
gamma_notAfrica = trace_7_5b['bR']


# #### Code 7.14

# In[25]:


print(f"Gamma within Africa: {gamma_Africa.mean():.2f}")
print(f"Gamma outside Africa: {gamma_notAfrica.mean():.2f}")


# #### Code 7.15

# In[26]:


_, ax = plt.subplots()
ax.set_xlabel('gamma')
ax.set_ylabel('Density')
ax.set_ylim(top=5.25)
az.plot_kde(gamma_Africa)
az.plot_kde(gamma_notAfrica, plot_kwargs={'color':'k'});


# #### Code 7.16

# In[27]:


diff = gamma_Africa - gamma_notAfrica
# First let's plot a histogram and a kernel densitiy estimate.
az.plot_kde(diff)
plt.hist(diff, bins=len(diff));
# Notice that there are very few values below zero.


# Hence the probability to have a negative slope association ruggedness with log-GDP inside Africa is so small, it might just be zero.

# In[28]:


sum(diff[diff < 0]) / len(diff)


# #### Code 7.17
# Plot the reverse interpretation: The influence of being in Africa depends upon terrain ruggedness.
# 
# This places `cont_africa` on the horizontal axis, while using different lines for different values of `rugged`.

# In[29]:


# Get min and max rugged values.
q_rugged = [0, 0]
q_rugged[0] = np.min(dd.rugged)
q_rugged[1] = np.max(dd.rugged)


# In[30]:


# Compute lines and confidence intervals.
# Since the link function isn't implemented we have to again compute the mean over samples ourselves using a loop.
mu_ruggedlo = np.zeros((2, len(trace_7_5b['bR'])))
mu_ruggedhi = np.zeros((2, len(trace_7_5b['bR'])))
# Iterate over outside Africa (0) and inside Africa (1).
for iAfri in range(0,2):
    mu_ruggedlo[iAfri] = trace_7_5b['a'] + trace_7_5b['bR'] * q_rugged[0] + \
                              trace_7_5b['bAR'] * q_rugged[0] * iAfri + \
                              trace_7_5b['bA'] * iAfri
    mu_ruggedhi[iAfri] = trace_7_5b['a'] + trace_7_5b['bR'] * q_rugged[1] + \
                              trace_7_5b['bAR'] * q_rugged[1] * iAfri + \
                              trace_7_5b['bA'] * iAfri


# In[31]:


mu_ruggedlo_mean = np.mean(mu_ruggedlo, axis=1)
mu_hpd_ruggedlo = pm.hpd(mu_ruggedlo.T, alpha=0.03)  # 97% probability interval: 1-.97 = 0.03
mu_ruggedhi_mean = np.mean(mu_ruggedhi, axis=1)
mu_hpd_ruggedhi = pm.hpd(mu_ruggedhi.T, alpha=0.03)  # 97% probability interval: 1-.97 = 0.03


# In[32]:


# Source http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))  # outward by 5 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


# In[33]:


# Plot it all, splitting points at median
med_r = np.median(dd.rugged)
# Use list comprehension to split points at median
ox = [0.05 if x > med_r else -0.05 for x in dd.rugged]
idxk = [i for i,x in enumerate(ox) if x == -0.05]
idxb = [i for i,x in enumerate(ox) if x == 0.05]
cont_africa_ox = dd.cont_africa + ox
plt.plot(cont_africa_ox[dd.cont_africa.index[idxk]], np.log(dd.rgdppc_2000[dd.cont_africa.index[idxk]]), 'ko')
plt.plot(cont_africa_ox[dd.cont_africa.index[idxb]], np.log(dd.rgdppc_2000[dd.cont_africa.index[idxb]]), 'C0o')
plt.plot([0, 1], mu_ruggedlo_mean, 'k--')
plt.plot([0, 1], mu_ruggedhi_mean, 'C0')
plt.fill_between([0, 1], mu_hpd_ruggedlo[:,0], mu_hpd_ruggedlo[:,1], color='k', alpha=0.2)
plt.fill_between([0, 1], mu_hpd_ruggedhi[:,0], mu_hpd_ruggedhi[:,1], color='b', alpha=0.2)
plt.ylabel('log GDP year 2000', fontsize=14);
plt.xlabel('Continent', fontsize=14)
axes = plt.gca()
axes.set_xlim([-0.25, 1.25])
axes.set_ylim([5.8, 11.2])
axes.set_xticks([0, 1])
axes.set_xticklabels(['other', 'Africa'], fontsize=12)
axes.set_facecolor('white')
adjust_spines(axes, ['left', 'bottom'])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(0.5)
axes.spines['left'].set_linewidth(0.5)
axes.spines['bottom'].set_color('black')
axes.spines['left'].set_color('black');


# #### Code 7.16

# In[34]:


d = pd.read_csv('Data/tulips.csv', sep=';', header=0)
d.info()
d.head()
d.describe()


# #### Code 7.19

# In[35]:


with pm.Model() as model_7_6:
    a = pm.Normal('a', mu=0, sd=100)
    bW = pm.Normal('bW', mu=0, sd=100)
    bS = pm.Normal('bS', mu=0, sd=100)
    sigma = pm.Uniform('sigma', lower=0, upper=100)
    mu = pm.Deterministic('mu', a + bW*d.water + bS*d.shade)
    blooms = pm.Normal('blooms', mu, sigma, observed=d.blooms)
    trace_7_6 = pm.sample(1000, tune=1000)


# In[36]:


with pm.Model() as model_7_7:
    a = pm.Normal('a', mu=0, sd=100)
    bW = pm.Normal('bW', mu=0, sd=100)
    bS = pm.Normal('bS', mu=0, sd=100)
    bWS = pm.Normal('bWS', mu=0, sd=100)
    sigma = pm.Uniform('sigma', lower=0, upper=100)
    mu = pm.Deterministic('mu', a + bW*d.water + bS*d.shade + bWS*d.water*d.shade)
    blooms = pm.Normal('blooms', mu, sigma, observed=d.blooms)
    trace_7_7 = pm.sample(1000, tune=1000)


# In[37]:


map_7_6 = pm.find_MAP(model=model_7_6)
map_7_6


# In[38]:


map_7_7 = pm.find_MAP(model=model_7_7)
map_7_7


# #### Code 7.20
# You can use the modified Powell's method if it fails with BFGS (default MAP estimate)

# In[39]:


from scipy import optimize

map_7_6 = pm.find_MAP(model=model_7_6, method='Powell')
map_7_6


# In[40]:


map_7_7 = pm.find_MAP(model=model_7_7, method='Powell')
map_7_7


# #### Code 7.21
# `conftab` is not implemented in PyMC, something similar is to use `summary()`

# In[41]:


az.summary(trace_7_6, var_names=['~mu'])['mean']


# In[42]:


az.summary(trace_7_7, var_names=['~mu'])['mean']


# #### Code 7.22

# In[43]:


comp_df = az.compare({'m7.6' : trace_7_6,
                      'm7.7' : trace_7_7})

comp_df


# #### 7.23
# Center and re-estimate

# In[44]:


d['shade_c'] = d.shade - np.mean(d.shade)
d['water_c'] = d.water - np.mean(d.water)


# #### 7.24
# No interaction.

# In[45]:


with pm.Model() as model_7_8:
    a = pm.Normal('a', mu=0, sd=100)
    bW = pm.Normal('bW', mu=0, sd=100)
    bS = pm.Normal('bS', mu=0, sd=100)
    sigma = pm.Uniform('sigma', lower=0, upper=100)
    mu = pm.Deterministic('mu', a + bW*d.water_c + bS*d.shade_c)
    blooms = pm.Normal('blooms', mu, sigma, observed=d.blooms)
    trace_7_8 = pm.sample(1000, tune=1000)
    start = {'a':np.mean(d.blooms), 'bW':0, 'bS':0, 'sigma':np.std(d.blooms)}


# Interaction.

# In[46]:


with pm.Model() as model_7_9:
    a = pm.Normal('a', mu=0, sd=100)
    bW = pm.Normal('bW', mu=0, sd=100)
    bS = pm.Normal('bS', mu=0, sd=100)
    bWS = pm.Normal('bWS', mu=0, sd=100)
    sigma = pm.Uniform('sigma', lower=0, upper=100)
    mu = pm.Deterministic('mu', a + bW*d.water_c + bS*d.shade_c + bWS*d.water_c*d.shade_c)
    blooms = pm.Normal('blooms', mu, sigma, observed=d.blooms)
    trace_7_9 = pm.sample(1000, tune=1000)
    start = {'a':np.mean(d.blooms), 'bW':0, 'bS':0, 'bWS':0, 'sigma':np.std(d.blooms)}


# In[47]:


map_7_8 = pm.find_MAP(model=model_7_8)
map_7_8


# In[48]:


map_7_9 = pm.find_MAP(model=model_7_9)
map_7_9


# #### 7.25

# In[49]:


map_7_7['a'] + map_7_7['bW'] * 2 + map_7_7['bS'] * 2 + map_7_7['bWS'] * 2 * 2


# #### 7.26

# In[50]:


map_7_9['a'] + map_7_9['bW'] * 0 + map_7_9['bS'] * 0 + map_7_9['bWS'] * 0 * 0


# #### 7.27

# In[51]:


varnames = ['a', 'bW', 'bS', 'bWS', 'sigma']
az.summary(trace_7_9, varnames, credible_interval=.89).round(3)


# #### 7.28
# 
# We have to replace the `link` function with a loop.

# In[52]:


# No interaction
f, axs = plt.subplots(1, 3, sharey=True, figsize=(8,3))
# Loop over values of water_c and plot predictions.
shade_seq = range(-1, 2, 1)

mu_w = np.zeros((len(shade_seq), len(trace_7_8['a'])))
for ax, w in zip(axs.flat, range(-1, 2, 1)):
    dt = d[d.water_c == w]
    ax.plot(dt.shade-np.mean(dt.shade), dt.blooms, 'C0o')
    for x, iSeq in enumerate(shade_seq):
        mu_w[x] = trace_7_8['a'] + trace_7_8['bW'] * w + trace_7_8['bS'] * iSeq
    mu_mean_w = mu_w.mean(1)
    mu_hpd_w = pm.hpd(mu_w.T, alpha=0.03)  # 97% probability interval: 1-.97 = 0.03
    ax.plot(shade_seq, mu_mean_w, 'k')
    ax.plot(shade_seq, mu_hpd_w.T[0], 'k--')
    ax.plot(shade_seq, mu_hpd_w.T[1], 'k--')
    ax.set_ylim(0,362)
    ax.set_ylabel('blooms')
    ax.set_xlabel('shade (centerd)')
    ax.set_title(f'water_c = {w:d}')
    ax.set_xticks(shade_seq)
    ax.set_yticks(range(0, 301, 100))

# Interaction
f, axs = plt.subplots(1, 3, sharey=True, figsize=(8,3))
# Loop over values of water_c and plot predictions.
shade_seq = range(-1, 2, 1)

mu_w = np.zeros((len(shade_seq), len(trace_7_9['a'])))
for ax, w in zip(axs.flat, range(-1, 2, 1)):
    dt = d[d.water_c == w]
    ax.plot(dt.shade-np.mean(dt.shade), dt.blooms, 'C0o')
    for x, iSeq in enumerate(shade_seq):
        mu_w[x] = trace_7_9['a'] + trace_7_9['bW'] * w + trace_7_9['bS'] * iSeq + trace_7_9['bWS'] * w * iSeq
    mu_mean_w = mu_w.mean(1)
    mu_hpd_w = az.hpd(mu_w.T, credible_interval=0.97)  # 97% probability interval: 1-.97 = 0.03
    ax.plot(shade_seq, mu_mean_w, 'k')
    ax.plot(shade_seq, mu_hpd_w.T[0], 'k--')
    ax.plot(shade_seq, mu_hpd_w.T[1], 'k--')
    ax.set_ylim(0,362)
    ax.set_ylabel('blooms')
    ax.set_xlabel('shade (centered)')
    ax.set_title(f'water_c = {w:d}')
    ax.set_xticks(shade_seq)
    ax.set_yticks(range(0, 301, 100))


# Let's remake the plots with water on abscissa while varying shade levels from left to right.

# In[53]:


# No interaction
f, axs = plt.subplots(1, 3, sharey=True, figsize=(8,3))
# Loop over values of water_c and plot predictions.
water_seq = range(-1, 2, 1)

mu_s = np.zeros((len(water_seq), len(trace_7_8['a'])))
for ax, s in zip(axs.flat, range(-1, 2, 1)):
    dt = d[d.shade_c == s]
    ax.plot(dt.water-np.mean(dt.water), dt.blooms, 'C0o')
    for x, iSeq in enumerate(shade_seq):
        mu_s[x] = trace_7_8['a'] + trace_7_8['bW'] * iSeq + trace_7_8['bS'] * s
    mu_mean_s = mu_s.mean(1)
    mu_hpd_s = pm.hpd(mu_s.T, alpha=0.03)  # 97% probability interval: 1-.97 = 0.03
    ax.plot(water_seq, mu_mean_s, 'k')
    ax.plot(water_seq, mu_hpd_s.T[0], 'k--')
    ax.plot(water_seq, mu_hpd_s.T[1], 'k--')
    ax.set_ylim(0,362)
    ax.set_ylabel('blooms')
    ax.set_xlabel('water (centerd)')
    ax.set_title(f'shade_c = {s:d}')
    ax.set_xticks(water_seq)
    ax.set_yticks(range(0, 301, 100))

# Interaction
f, axs = plt.subplots(1, 3, sharey=True, figsize=(8,3))
# Loop over values of water_c and plot predictions.
water_seq = range(-1, 2, 1)

mu_s = np.zeros((len(water_seq), len(trace_7_9['a'])))
for ax, s in zip(axs.flat, range(-1, 2, 1)):
    dt = d[d.shade_c == s]
    ax.plot(dt.water-np.mean(dt.water), dt.blooms, 'C0o')
    for x, iSeq in enumerate(water_seq):
        mu_s[x] = trace_7_9['a'] + trace_7_9['bW'] * iSeq + trace_7_9['bS'] * s + trace_7_9['bWS'] * iSeq * s
    mu_mean_s = mu_s.mean(1)
    mu_hpd_s = az.hpd(mu_s.T, credible_interval=.97)  # 97% probability interval: 1-.97 = 0.03
    ax.plot(water_seq, mu_mean_s, 'k')
    ax.plot(water_seq, mu_hpd_s.T[0], 'k--')
    ax.plot(water_seq, mu_hpd_s.T[1], 'k--')
    ax.set_ylim(0,362)
    ax.set_ylabel('blooms')
    ax.set_xlabel('water (centerd)')
    ax.set_title(f'shade_c = {s:d}')
    ax.set_xticks(water_seq)
    ax.set_yticks(range(0, 301, 100))


# When there is no interaction the slope is the same across all three plots (top row), showing a general reduction with increasing shade. For the interaction (bottom row) we can see a huge increase in blooms for the lowest amount of shade as we increase water. This effect is reduced by increasing shade to average levels and in the last plot increasing water has a minimum effect when there is lots of shade.

# #### 7.29

# In[54]:


m_7_x = smf.ols('blooms ~ shade + water + shade * water', data=d).fit()


# #### 7.30

# In[55]:


m_7_x = smf.ols('blooms ~ shade * water', data=d).fit()


# #### 7.31

# In[56]:


m_7_x = smf.ols('blooms ~ shade * water - water', data=d).fit()


# #### 7.32

# In[57]:


m_7_x = smf.ols('blooms ~ shade * water * bed', data=d).fit()


# #### 7.33
# Not sure how this one works

# In[58]:


from patsy import dmatrix

x, y, z = 1, 1, 1
d_matrix = dmatrix('~ x * y * w')
d_matrix.design_info.column_names 


# In[59]:


import platform
import sys

import IPython
import matplotlib
import scipy

print("""This notebook was created using:\nPython {}\nIPython {}\nPyMC {}\nArviZ {}\nNumPy {}\nSciPy {}\nMatplotlib {}\n""".format(sys.version[:5], IPython.__version__, pm.__version__, az.__version__, np.__version__, scipy.__version__, matplotlib.__version__))


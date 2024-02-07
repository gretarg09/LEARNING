#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy as sp

from theano import shared


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use('arviz-darkgrid')


# #### Code 11.1

# In[3]:


trolley_df = pd.read_csv('Data/Trolley.csv', sep=';')
trolley_df.head()


# #### Code 11.2

# In[4]:


ax = (trolley_df.response
                .value_counts()
                .sort_index()
                .plot(kind='bar'))

ax.set_xlabel("response", fontsize=14);
ax.set_ylabel("Frequency", fontsize=14);


# #### Code 11.3

# In[5]:


ax = (trolley_df.response
                .value_counts()
                .sort_index()
                .cumsum()
                .div(trolley_df.shape[0])
                .plot(marker='o'))

ax.set_xlim(0.9, 7.1);
ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("cumulative proportion", fontsize=14);


# #### Code 11.4

# In[6]:


resp_lco = (trolley_df.response
                      .value_counts()
                      .sort_index()
                      .cumsum()
                      .iloc[:-1]
                      .div(trolley_df.shape[0])
                      .apply(lambda p: np.log(p / (1. - p))))


# In[7]:


ax = resp_lco.plot(marker='o')

ax.set_xlim(0.9, 7);
ax.set_xlabel("response", fontsize=14)
ax.set_ylabel("log-cumulative-odds", fontsize=14);


# #### Code 11.5

# In[8]:


with pm.Model() as m11_1:
    a = pm.Normal(
        'a', 0., 10.,
        transform=pm.distributions.transforms.ordered,
        shape=6, testval=np.arange(6) - 2.5)
    
    resp_obs = pm.OrderedLogistic(
        'resp_obs', 0., a,
        observed=trolley_df.response.values - 1
    )


# In[9]:


with m11_1:
    map_11_1 = pm.find_MAP()


# #### Code 11.6

# In[10]:


map_11_1['a']


# In[11]:


daf


# #### Code 11.7

# In[ ]:


sp.special.expit(map_11_1['a'])


# #### Code 11.8

# In[ ]:


with m11_1:
    trace_11_1 = pm.sample(1000, tune=1000)


# In[ ]:


az.summary(trace_11_1, var_names=['a'], credible_interval=.89, rount_to=2)


# #### Code 11.9

# In[ ]:


def ordered_logistic_proba(a):
    pa = sp.special.expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    
    return p_cum[1:] - p_cum[:-1]


# In[ ]:


ordered_logistic_proba(trace_11_1['a'].mean(axis=0))


# #### Code 11.10

# In[ ]:


(ordered_logistic_proba(trace_11_1['a'].mean(axis=0)) \
     * (1 + np.arange(7))).sum()


# #### Code 11.11

# In[ ]:


ordered_logistic_proba(trace_11_1['a'].mean(axis=0) - 0.5)


# #### Code 11.12

# In[ ]:


(ordered_logistic_proba(trace_11_1['a'].mean(axis=0) - 0.5) \
     * (1 + np.arange(7))).sum()


# #### Code 11.13

# In[ ]:


action = shared(trolley_df.action.values)
intention = shared(trolley_df.intention.values)
contact = shared(trolley_df.contact.values)

with pm.Model() as m11_2:
    a = pm.Normal(
        'a', 0., 10.,
        transform=pm.distributions.transforms.ordered,
        shape=6,
        testval=trace_11_1['a'].mean(axis=0)
    )
    
    bA = pm.Normal('bA', 0., 10.)
    bI = pm.Normal('bI', 0., 10.)
    bC = pm.Normal('bC', 0., 10.)
    phi = bA * action + bI * intention + bC * contact

    resp_obs = pm.OrderedLogistic(
        'resp_obs', phi, a,
        observed=trolley_df.response.values - 1
    )


# In[ ]:


with m11_2:
    map_11_2 = pm.find_MAP()


# #### Code 11.14

# In[ ]:


with pm.Model() as m11_3:
    a = pm.Normal(
        'a', 0., 10.,
        transform=pm.distributions.transforms.ordered,
        shape=6,
        testval=trace_11_1['a'].mean(axis=0)
    )
    
    bA = pm.Normal('bA', 0., 10.)
    bI = pm.Normal('bI', 0., 10.)
    bC = pm.Normal('bC', 0., 10.)
    bAI = pm.Normal('bAI', 0., 10.)
    bCI = pm.Normal('bCI', 0., 10.)
    phi  = bA * action + bI * intention + bC * contact \
            + bAI * action * intention \
            + bCI * contact * intention

    resp_obs = pm.OrderedLogistic(
        'resp_obs', phi, a,
        observed=trolley_df.response - 1
    )


# In[ ]:


with m11_3:
    map_11_3 = pm.find_MAP()


# #### Code 11.15

# In[ ]:


def get_coefs(map_est):
    coefs = OrderedDict()
    
    for i, ai in enumerate(map_est['a']):
        coefs[f'a_{i}'] = ai
        
    coefs['bA'] = map_est.get('bA', np.nan)
    coefs['bI'] = map_est.get('bI', np.nan)
    coefs['bC'] = map_est.get('bC', np.nan)
    coefs['bAI'] = map_est.get('bAI', np.nan)
    coefs['bCI'] = map_est.get('bCI', np.nan)
        
    return coefs


# In[ ]:


(pd.DataFrame.from_dict(
    OrderedDict([
        ('m11_1', get_coefs(map_11_1)),
        ('m11_2', get_coefs(map_11_2)),
        ('m11_3', get_coefs(map_11_3))
    ]))
   .astype(np.float64)
   .round(2))


# #### Code 11.16

# In[ ]:


with m11_2:
    trace_11_2 = pm.sample(1000, tune=1000)


# In[ ]:


with m11_3:
    trace_11_3 = pm.sample(1000, tune=1000)


# In[ ]:


comp_df = pm.compare({m11_1:trace_11_1,
                      m11_2:trace_11_2,
                      m11_3:trace_11_3})

comp_df.loc[:,'model'] = pd.Series(['m11.1', 'm11.2', 'm11.3'])
comp_df = comp_df.set_index('model')
comp_df


# #### Code 11.17-19

# In[ ]:


pp_df = pd.DataFrame(np.array([[0, 0, 0],
                               [0, 0, 1],
                               [1, 0, 0],
                               [1, 0, 1],
                               [0, 1, 0],
                               [0, 1, 1]]),
                     columns=['action', 'contact', 'intention'])


# In[ ]:


pp_df


# In[ ]:


action.set_value(pp_df.action.values)
contact.set_value(pp_df.contact.values)
intention.set_value(pp_df.intention.values)

with m11_3:
    pp_trace_11_3 = pm.sample_ppc(trace_11_3, samples=1500)


# In[ ]:


PP_COLS = [f'pp_{i}' for i, _ in enumerate(pp_trace_11_3['resp_obs'])]

pp_df = pd.concat((pp_df,
                   pd.DataFrame(pp_trace_11_3['resp_obs'].T, columns=PP_COLS)),
                  axis=1)


# In[ ]:


pp_cum_df = (pd.melt(
                    pp_df,
                    id_vars=['action', 'contact', 'intention'],
                    value_vars=PP_COLS, value_name='resp'
                )
               .groupby(['action', 'contact', 'intention', 'resp'])
               .size()
               .div(1500)
               .rename('proba')
               .reset_index()
               .pivot_table(
                   index=['action', 'contact', 'intention'],
                   values='proba',
                   columns='resp'
                )
               .cumsum(axis=1)
               .iloc[:, :-1])


# In[ ]:


pp_cum_df


# In[ ]:


for (plot_action, plot_contact), plot_df in pp_cum_df.groupby(level=['action', 'contact']):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], plot_df, c='C0');
    ax.plot([0, 1], [0, 0], '--', c='C0');
    ax.plot([0, 1], [1, 1], '--', c='C0');
    
    ax.set_xlim(0, 1);
    ax.set_xlabel("intention");
    
    ax.set_ylim(-0.05, 1.05);
    ax.set_ylabel("probability");
    
    ax.set_title(
        "action = {action}, contact = {contact}".format(
            action=plot_action, contact=plot_contact
        )
    );


# #### Code 11.20

# In[ ]:


# define parameters
PROB_DRINK = 0.2 # 20% of days
RATE_WORK = 1. # average 1 manuscript per day

# sample one year of production
N = 365


# In[ ]:


drink = np.random.binomial(1, PROB_DRINK, size=N)
y = (1 - drink) * np.random.poisson(RATE_WORK, size=N)


# #### Code 11.21

# In[ ]:


drink_zeros = drink.sum()
work_zeros = (y == 0).sum() - drink_zeros


# In[ ]:


bins = np.arange(y.max() + 1) - 0.5

plt.hist(y, bins=bins);
plt.bar(0., drink_zeros, width=1., bottom=work_zeros, color='C1', alpha=.5);

plt.xticks(bins + 0.5);
plt.xlabel("manuscripts completed");

plt.ylabel("Frequency");


# #### Code 11.22

# In[ ]:


with pm.Model() as m11_4:
    ap = pm.Normal('ap', 0., 1.)
    p = pm.math.sigmoid(ap)
    
    al = pm.Normal('al', 0., 10.)
    lambda_ = pm.math.exp(al)
    
    y_obs = pm.ZeroInflatedPoisson('y_obs', 1. - p, lambda_, observed=y)


# In[ ]:


with m11_4:
    map_11_4 = pm.find_MAP()


# In[ ]:


map_11_4


# #### Code 11.23

# In[ ]:


sp.special.expit(map_11_4['ap']) # probability drink


# In[ ]:


np.exp(map_11_4['al']) # rate finish manuscripts, when not drinking


# #### Code 11.24

# In[ ]:


def dzip(x, p, lambda_, log=True):
    like = p**(x == 0) + (1 - p) * sp.stats.poisson.pmf(x, lambda_)
    
    return np.log(like) if log else like


# #### Code 11.25

# In[ ]:


PBAR = 0.5
THETA = 5.


# In[ ]:


a = PBAR * THETA
b = (1 - PBAR) * THETA


# In[ ]:


p = np.linspace(0, 1, 100)

plt.plot(p, sp.stats.beta.pdf(p, a, b));

plt.xlim(0, 1);
plt.xlabel("probability");

plt.ylabel("Density");


# #### Code 11.26

# In[ ]:


admit_df = pd.read_csv('Data/UCBadmit.csv', sep=';')
admit_df.head()


# In[ ]:


with pm.Model() as m11_5:
    a = pm.Normal('a', 0., 2.)
    pbar = pm.Deterministic('pbar', pm.math.sigmoid(a))

    theta = pm.Exponential('theta', 1.)
    
    admit_obs = pm.BetaBinomial(
        'admit_obs',
        pbar * theta, (1. - pbar) * theta,
        admit_df.applications.values,
        observed=admit_df.admit.values
    )


# In[ ]:


with m11_5:
    trace_11_5 = pm.sample(1000, tune=1000)


# #### Code 11.27

# In[ ]:


pm.summary(trace_11_5, alpha=.11).round(2)


# #### Code 11.28

# In[ ]:


np.percentile(trace_11_5['pbar'], [2.5, 50., 97.5])


# #### Code 11.29

# In[ ]:


pbar_hat = trace_11_5['pbar'].mean()
theta_hat = trace_11_5['theta'].mean()

p_plot = np.linspace(0, 1, 100)

plt.plot(
    p_plot,
    sp.stats.beta.pdf(p_plot, pbar_hat * theta_hat, (1. - pbar_hat) * theta_hat)
);
plt.plot(
    p_plot,
    sp.stats.beta.pdf(
        p_plot[:, np.newaxis],
        trace_11_5['pbar'][:100] * trace_11_5['theta'][:100],
        (1. - trace_11_5['pbar'][:100]) * trace_11_5['theta'][:100]
    ),
    c='C0', alpha=0.1
);

plt.xlim(0., 1.);
plt.xlabel("probability admit");

plt.ylim(0., 3.);
plt.ylabel("Density");


# #### Code 11.30

# In[ ]:


with m11_5:
    pp_trace_11_5 = pm.sample_ppc(trace_11_5)


# In[ ]:


x_case = np.arange(admit_df.shape[0])

plt.scatter(
    x_case,
    pp_trace_11_5['admit_obs'].mean(axis=0) \
        / admit_df.applications.values
);
plt.scatter(x_case, admit_df.admit / admit_df.applications);

high = np.percentile(pp_trace_11_5['admit_obs'], 95, axis=0) \
        / admit_df.applications.values
plt.scatter(x_case, high, marker='x', c='k');

low = np.percentile(pp_trace_11_5['admit_obs'], 5, axis=0) \
        / admit_df.applications.values
plt.scatter(x_case, low, marker='x', c='k');


# #### Code 11.31

# In[ ]:


mu = 3.
theta = 1.

x = np.linspace(0, 10, 100)
plt.plot(x, sp.stats.gamma.pdf(x, mu / theta, scale=theta));


# In[ ]:


import platform
import sys

import IPython
import matplotlib
import scipy

print("This notebook was createad on a computer {} running {} and using:\nPython {}\nIPython {}\nPyMC {}\nNumPy {}\nPandas {}\nSciPy {}\nMatplotlib {}\n".format(platform.machine(), ' '.join(platform.linux_distribution()[:2]), sys.version[:5], IPython.__version__, pm.__version__, np.__version__, pd.__version__, scipy.__version__, matplotlib.__version__))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import statsmodels.formula.api as smf

from scipy import stats


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use('arviz-darkgrid')


# #### Code 5.1

# In[3]:


# load data
d = pd.read_csv('Data/WaffleDivorce.csv', sep=';')
# standardize predictor
d['MedianAgeMarriage_s'] = (d.MedianAgeMarriage - d.MedianAgeMarriage.mean()) / d.MedianAgeMarriage.std()


# In[4]:


with pm.Model() as model_5_1:
    a = pm.Normal('a', mu=10, sd=10)
    bA = pm.Normal('bA', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    # good (default) alternatives for sigma (in this and other models) are
    # sigma = pm.HalfNormal('sigma', 5)
    # sigma = pm.HalfCauchy('sigma', 5)
    # some people recomed avoiding "hard" boundaries unless they have a theoretical/data-based justification, like a correlation that is restricted to be [-1, 1].
    mu = pm.Deterministic('mu', a + bA * d.MedianAgeMarriage_s)
    Divorce = pm.Normal('Divorce', mu=mu, sd=sigma, observed=d.Divorce)
    trace_5_1 = pm.sample(1000, tune=1000)


# In[5]:


varnames = ['~mu']
az.plot_trace(trace_5_1, varnames);


# #### Code 5.2

# In[6]:


mu_mean = trace_5_1['mu']

plt.plot(d.MedianAgeMarriage_s, d.Divorce, 'C0o')
plt.plot(d.MedianAgeMarriage_s, mu_mean.mean(0), 'C1')

az.plot_hpd(d.MedianAgeMarriage_s, mu_mean)

plt.xlabel('MedianAgeMarriage_s')
plt.ylabel('Divorce');


# #### Code 5.3

# In[7]:


d['Marriage_s'] = (d.Marriage - d.Marriage.mean()) / d.Marriage.std()


# In[8]:


with pm.Model() as model_5_2:
    a = pm.Normal('a', mu=10, sd=10)
    bA = pm.Normal('bA', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bA * d.Marriage_s)
    Divorce = pm.Normal('Divorce', mu=mu, sd=sigma, observed=d.Divorce)
    trace_5_2 = pm.sample(1000, tune=1000)


# In[9]:


az.plot_trace(trace_5_2, varnames);


# In[10]:


mu_mean = trace_5_2['mu']

d.plot('Marriage_s', 'Divorce', kind='scatter', xlim = (-2, 3))
plt.plot(d.Marriage_s, mu_mean.mean(0), 'C1')

az.plot_hpd(d.Marriage_s, mu_mean);


# #### Code 5.4

# In[11]:


with pm.Model() as model_5_3:
    a = pm.Normal('a', mu=10, sd=10)
    bA = pm.Normal('bA', mu=0, sd=1, shape=2)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bA[0] * d.Marriage_s + bA[1] * d.MedianAgeMarriage_s)
    
    Divorce = pm.Normal('Divorce', mu=mu, sd=sigma, observed=d.Divorce)
    trace_5_3 = pm.sample(1000, tune=1000)


# In[12]:


az.plot_trace(trace_5_3, varnames);


# In[13]:


az.summary(trace_5_3, varnames, credible_interval=.11).round(3)


# #### Code 5.5

# In[14]:


pm.forestplot(trace_5_3, var_names=varnames);


# #### Code 5.6

# In[15]:


with pm.Model() as model_5_4:
    a = pm.Normal('a', mu=10, sd=10)
    b = pm.Normal('b', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + b * d.MedianAgeMarriage_s)
    Marriage = pm.Normal('Marriage', mu=mu, sd=sigma, observed=d.Marriage_s)
    trace_5_4 = pm.sample(1000, tune=1000)


# In[16]:


pm.traceplot(trace_5_4, varnames);


# #### Code 5.7

# In[17]:


mu_pred = trace_5_4['mu'].mean(0)
residuals = d.Marriage_s - mu_pred


# #### Code 5.8

# In[18]:


idx = np.argsort(d.MedianAgeMarriage_s)
d.plot('MedianAgeMarriage_s', 'Marriage_s', kind='scatter', xlim = (-3, 3), ylim = (-3, 3))
plt.plot(d.MedianAgeMarriage_s[idx], mu_pred[idx], 'k')
plt.vlines(d.MedianAgeMarriage_s, mu_pred, mu_pred + residuals, colors='grey');


# #### Code 5.9

# In[19]:


R_avg = np.linspace(-3, 3, 100)

mu_pred = trace_5_3['a'] + trace_5_3['bA'][:,0] * R_avg[:,None]
divorce_ = stats.norm.rvs(mu_pred, trace_5_3['sigma'])

plt.plot(R_avg, mu_pred.mean(1), 'C0')
az.plot_hpd(R_avg, mu_pred.T)
az.plot_hpd(R_avg, divorce_.T)

plt.xlabel('Marriage_s')
plt.ylabel('Divorce')
plt.title('MedianAgeMarriage_s = 0');


# #### Code 5.10

# In[20]:


R_avg = np.linspace(-3, 3, 100)

mu_pred = trace_5_3['a'] + trace_5_3['bA'][:,1] * R_avg[:,None]
divorce_ = stats.norm.rvs(mu_pred, trace_5_3['sigma'])

plt.plot(R_avg, mu_pred.mean(1), 'C0')
az.plot_hpd(R_avg, mu_pred.T)
az.plot_hpd(R_avg, divorce_.T)

plt.xlabel('MedianAgeMarriage_s')
plt.ylabel('Divorce')
plt.title('Marriage_s = 0');


# #### Code 5.11

# In[21]:


mu_pred = trace_5_3['mu']

divorce_pred = pm.sample_posterior_predictive(trace_5_3, samples=1000, model=model_5_3)['Divorce']
divorce_hpd = pm.hpd(divorce_pred)


# #### Code 5.12

# In[22]:


mu_hpd = az.hpd(mu_pred, credible_interval=0.95)
plt.errorbar(d.Divorce, divorce_pred.mean(0), yerr=np.abs(divorce_pred.mean(0)-mu_hpd.T) , fmt='C0o')
plt.plot(d.Divorce, divorce_pred.mean(0), 'C0o')

plt.xlabel('Observed divorce', fontsize=14)
plt.ylabel('Predicted divorce', fontsize=14)

min_x, max_x = d.Divorce.min(), d.Divorce.max()
plt.plot([min_x, max_x], [min_x, max_x], 'k--');


# #### Code 5.14

# In[23]:


plt.figure(figsize=(10,12))
residuals = d.Divorce - mu_pred.mean(0)
idx = np.argsort(residuals)
y_label = d.Loc[idx]
y_points = np.linspace(0, 1, 50)
plt.errorbar(residuals[idx], y_points, 
             xerr=np.abs(divorce_pred.mean(0)-mu_hpd.T),
             fmt='C0o',lw=3)

plt.errorbar(residuals[idx], y_points, 
             xerr=np.abs(divorce_pred.mean(0)-divorce_hpd.T),
             fmt='C0o', lw=3, alpha=0.5)

plt.yticks(y_points, y_label);
plt.vlines(0, 0, 1, 'grey');


# #### Code 5.15

# In[24]:


N = 100
x_real = stats.norm.rvs(size=N)
x_spur = stats.norm.rvs(x_real)
y = stats.norm.rvs(x_real)
d = pd.DataFrame([y, x_real, x_spur]).T

sns.pairplot(d);


# #### Code 5.16

# In[25]:


d = pd.read_csv('Data/milk.csv', sep=';')
d.head()


# #### Code 5.17 to 5.20

# In[26]:


dcc = d.dropna().copy()


# In[27]:


with pm.Model() as model_5_5:
    a = pm.Normal('a', mu=10, sd=100)
    bn = pm.Normal('bn', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=1)
    mu = pm.Deterministic('mu', a + bn * dcc['neocortex.perc'])
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=dcc['kcal.per.g'])
    trace_5_5 = pm.sample(1000, tune=1000)


# In[28]:


pm.traceplot(trace_5_5, varnames);


# #### Code 5.21

# In[29]:


az.summary(trace_5_5, varnames, credible_interval=.89).round(3)


# #### Code 5.22

# In[30]:


trace_5_5['bn'].mean() * (76 - 55)


# #### Code 5.23

# In[31]:


seq = np.linspace(55, 76, 50)
mu_pred = trace_5_5['a'] + trace_5_5['bn'] * seq[:,None]

plt.plot(d['neocortex.perc'], d['kcal.per.g'], 'C0o')

plt.plot(seq, mu_pred.mean(1), 'k')
az.plot_hpd(seq, mu_pred.T,
            fill_kwargs={'alpha': 0},
            plot_kwargs={'alpha':1, 'color':'k', 'ls':'--'})

plt.xlabel('neocortex.perc')
plt.ylabel('kcal.per.g');


# #### Code 5.24

# In[32]:


dcc['log_mass'] = np.log(dcc['mass'])


# #### Code 5.25

# In[33]:


with pm.Model() as model_5_6:
    a = pm.Normal('a', mu=10, sd=100)
    bn = pm.Normal('bn', mu=0, sd=1)
    sigma = pm.Uniform('sigma', lower=0, upper=1)
    mu = pm.Deterministic('mu', a + bn * dcc['log_mass'])
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=dcc['kcal.per.g'])
    trace_5_6 = pm.sample(1000, tune=1000)


# In[34]:


az.summary(trace_5_6, varnames, credible_interval=.89).round(3)


# #### Code 5.26

# In[35]:


with pm.Model() as model_5_7:
    a = pm.Normal('a', mu=10, sd=100)
    bn = pm.Normal('bn', mu=0, sd=1, shape=2)
    sigma = pm.Uniform('sigma', lower=0, upper=1)
    mu = pm.Deterministic('mu', a + bn[0] * dcc['neocortex.perc'] + bn[1] * dcc['log_mass'])
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=dcc['kcal.per.g'])
    trace_5_7 = pm.sample(1000, tune=1000)


# In[36]:


az.summary(trace_5_7, varnames, credible_interval=.89).round(3)


# #### Code 5.27

# In[37]:


seq = np.linspace(55, 76, 50)
mu_pred = trace_5_7['a'] + trace_5_7['bn'][:,0] * seq[:,None] + trace_5_7['bn'][:,1] * dcc['log_mass'].mean()

plt.plot(seq, mu_pred.mean(1), 'k')
az.plot_hpd(seq, mu_pred.T,
            fill_kwargs={'alpha': 0},
            plot_kwargs={'alpha':1, 'color':'k', 'ls':'--'})

plt.xlabel('neocortex.perc')
plt.ylabel('kcal.per.g');


# #### Code 5.28

# In[38]:


N = 100
rho = 0.7
x_pos = stats.norm.rvs(size=N)
x_neg = stats.norm.rvs(rho*x_pos, (1-rho**2)**0.5)
y = stats.norm.rvs(x_pos - x_neg)
d = pd.DataFrame([y, x_real, x_spur]).T

sns.pairplot(d);


# #### Code 5.29

# In[39]:


N = 100
height = stats.norm.rvs(size=N, loc=10, scale=2)
leg_prop = stats.uniform.rvs(size=N, loc=0.4, scale=0.5) 
leg_left = leg_prop * height + stats.norm.rvs(size=N, loc=0, scale=0.02)
leg_right = leg_prop * height + stats.norm.rvs(size=N, loc=0, scale=0.02)


# #### Code 5.30

# In[40]:


with pm.Model() as m5_8:
    a = pm.Normal('a', mu=10, sd=100)
    bl = pm.Normal('bl', mu=2, sd=10)
    br = pm.Normal('br', mu=2, sd=10)
    mu = pm.Deterministic('mu', a + bl * leg_left + br * leg_right)    
    sigma = pm.Uniform('sigma', lower=0 , upper=10)
    height_p = pm.Normal('height_p', mu=mu, sd=sigma, observed=height)
    
    trace_5_8 = pm.sample(1000, tune=1000)


# In[41]:


az.plot_trace(trace_5_8, varnames);
az.summary(trace_5_8, varnames, credible_interval=.89).round(3)


# #### Code 5.31

# In[42]:


az.plot_forest(trace_5_8, var_names=varnames);


# #### Code 5.32

# In[43]:


plt.scatter(trace_5_8['br'], trace_5_8['bl']);


# #### Code 5.33

# In[44]:


sum_blbr = trace_5_8['br'] + trace_5_8['bl']
az.plot_kde(sum_blbr);


# #### Code 5.34

# In[45]:


with pm.Model() as m5_9:
    a = pm.Normal('a',mu = 10, sd=100)
    bl = pm.Normal('bl',mu=2, sd= 10)
    mu = pm.Deterministic('mu',a + bl * leg_left)    
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    height = pm.Normal('height',mu=mu, sd=sigma, observed=height)
    trace_5_9 = pm.sample(1000, tune=1000)


# In[46]:


#pm.traceplot(trace_5_9, varnames)
az.summary(trace_5_9, varnames, credible_interval=.11).round(3)


# #### Code 5.35

# In[47]:


milk = pd.read_csv('Data/milk.csv', sep=';')
milk.head()


# #### Code 5.36

# In[48]:


with pm.Model() as m5_10:
    a = pm.Normal('a',mu = 0.6, sd=10)
    bf = pm.Normal('bf',mu=0, sd= 1)
    mu = pm.Deterministic('mu',a + bf * milk['perc.fat'])    
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    kcalperg = pm.Normal('kcal.per.g',mu=mu, sd=sigma, observed=milk['kcal.per.g'])
    trace_5_10 = pm.sample(1000, tune=1000)


# In[49]:


az.summary(trace_5_10, varnames, credible_interval=.89).round(3)


# In[50]:


with pm.Model() as m5_11:
    a = pm.Normal('a',mu = 0.6, sd=10)
    bl = pm.Normal('bl',mu=0, sd= 1)
    mu = pm.Deterministic('mu',a + bl * milk['perc.lactose'])    
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    kcalperg = pm.Normal('kcal.per.g',mu=mu, sd=sigma, observed=milk['kcal.per.g'])
    trace_5_11 = pm.sample(1000, tune=1000)


# In[51]:


az.summary(trace_5_11, varnames, credible_interval=.89).round(3)


# #### Code 5.37

# In[52]:


with pm.Model() as m5_12:
    a = pm.Normal('a',mu = 0.6, sd=10)
    bf = pm.Normal('bf',mu=0, sd= 1)
    bl = pm.Normal('bl',mu=0, sd= 1)
    mu = pm.Deterministic('mu',a + bf * milk['perc.fat'] + bl * milk['perc.lactose'])    
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    kcalperg = pm.Normal('kcal.per.g',mu=mu, sd=sigma, observed=milk['kcal.per.g'])
    trace_5_12 = pm.sample(1000, tune=1000)


# In[53]:


az.summary(trace_5_12, varnames, credible_interval=.89).round(3)


# #### Code 5.38

# In[54]:


df = milk[['kcal.per.g','perc.fat','perc.lactose']]
sns.pairplot(df);


# #### Code 5.39

# In[55]:


milk.corr()['perc.fat']['perc.lactose']


# #### Code 5.40

# In[56]:


def simcoll(r = 0.9):
    milk['x'] = stats.norm.rvs(size=len(milk), loc = r * milk['perc.fat'],
                       scale = np.sqrt((1 - r**2) * milk['perc.fat'].var()))
    X = np.column_stack((milk['perc.fat'], milk['x']))
    m = smf.OLS(milk['kcal.per.g'], X).fit()
    cov = m.cov_params()
    return (np.diag(cov)[1])**0.5
    
def repsimcoll(r= 0.9, N = 100):
    stddev = [simcoll(r) for _ in range(N)]
    return np.mean(stddev)

lista = []
for i in np.arange(start = 0, stop = 0.99, step = 0.01):   
    lista.append(repsimcoll (r= i, N = 100))

plt.plot(np.arange(start = 0, stop = 0.99, step = 0.01), lista)
plt.xlabel('correlation', fontsize=14)
plt.ylabel('stddev', fontsize=14);


# #### Code 5.41

# In[57]:


# number of plants
N = 100

# simulate initial heights
h0 = stats.norm.rvs(size = N, loc = 10, scale = 2)

# assign treatments and simulate fungus and growth
treatment = np.repeat([0, 1], [N/2]*2)
fungus = np.random.binomial(n=1, p=(0.5-treatment * 0.4), size=N)
h1 = h0 + stats.norm.rvs(size= N, loc= 5- 3*fungus, scale=1)

# compose a clean data frame
d = pd.DataFrame({'h0': h0,
                  'h1': h1,
                  'Treatment':treatment,
                  'Fungus': fungus})


# #### Code 5.42

# In[58]:


with pm.Model() as m5_13:
    a = pm.Normal('a',mu = 0, sd=100)
    bh = pm.Normal('bh',mu = 0, sd=10)
    bt = pm.Normal('bt',mu = 0, sd=10)
    bf = pm.Normal('bf',mu = 0, sd=10)
    mu = pm.Deterministic('mu',a + bh * h0 + bt * treatment + bf * fungus)
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    h1 = pm.Normal('h1', mu = mu, sd=sigma, observed = d['h1'].get_values())
    trace_5_13 = pm.sample(1000, tune=1000) 


# In[59]:


az.summary(trace_5_13, varnames, credible_interval=.89).round(3)


# #### Code 5.43

# In[60]:


with pm.Model() as m5_14:
    a = pm.Normal('a',mu = 0, sd=100)
    bh = pm.Normal('bh',mu = 0, sd=10)
    bt = pm.Normal('bt',mu = 0, sd=10)
    mu = pm.Deterministic('mu',a + bh * h0 + bt * treatment)
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    h1 = pm.Normal('h1', mu = mu, sd=sigma, observed =d['h1'])
    trace_5_14 = pm.sample(1000, tune=1000) 


# In[61]:


az.summary(trace_5_14, varnames, credible_interval=.89).round(3)


# #### Code 5.44

# In[62]:


d = pd.read_csv('Data/Howell1.csv', sep=';')
d.head()


# #### Code 5.45

# In[63]:


with pm.Model() as m5_15:
    a = pm.Normal('a',mu = 178, sd=100)
    bm = pm.Normal('bm',mu = 0, sd=10)
    mu = pm.Deterministic('mu',a + bm * d['male'])
    sigma = pm.Uniform('sigma', lower= 0 , upper= 50)
    height = pm.Normal('height', mu = mu, sd=sigma, observed = d['height'])
    trace_5_15 = pm.sample(1000, tune=1000) 


# In[64]:


az.summary(trace_5_15, varnames, credible_interval=.89).round(3)


# #### Code 5.46

# In[65]:


mu.male = trace_5_15['a'] + trace_5_15['bm']
pm.hpd(mu.male)


# #### Code 5.47

# In[66]:


with pm.Model() as m5_15b:
    af = pm.Normal('af',mu = 178, sd=100)
    am = pm.Normal('am',mu = 178, sd=100)
    mu = pm.Deterministic('mu',af * (1 - d['male']) + am * d['male'])
    sigma = pm.Uniform('sigma', lower= 0 , upper= 50)
    height = pm.Normal('height', mu = mu, sd=sigma, observed = d['height'])
    trace_5_15b = pm.sample(1000, tune=1000) 


# #### Code 5.48

# In[67]:


d = pd.read_csv('Data/milk.csv', sep=';')
d = d.drop_duplicates()


# #### Code 5.49

# In[68]:


d['clade.NWM'] = np.where( d['clade'] == 'New World Monkey', 1, 0)
d['clade.NWM'].get_values()


# #### Code 5.50

# In[69]:


d['clade.OWM'] = np.where( d['clade'] == 'Old World Monkey', 1, 0)
d['clade.S'] = np.where( d['clade'] == 'Strepsirrhine', 1, 0)


# #### Code 5.51

# In[70]:


with pm.Model() as m5_16:
    a = pm.Normal('a', mu = 0.6, sd=10)
    b_NWM = pm.Normal('b_NWM',mu = 0, sd=1)
    b_OWM = pm.Normal('b_OWM',mu = 0, sd=1)
    b_S = pm.Normal('b_S',mu = 0, sd=1)
    mu = pm.Deterministic('mu', a + 
                          b_NWM * d['clade.NWM'] + 
                          b_OWM * d['clade.OWM'] +
                          b_S * d['clade.S'])
    
    # instead of adding this as a deterministic when running the model 
    # it is possible to add them, after sampling using something like this
    # trace_5_16.add_values({'mu_NWM', trace_5_16[a] + trace_5_16['b_NWM']})
    mu_ape = pm.Deterministic('mu_ape', a + 0)
    mu_NWM = pm.Deterministic('mu_NWM', a + b_NWM)
    mu_OWM = pm.Deterministic('mu_OWM', a + b_OWM)
    mu_S =  pm.Deterministic('mu_S', a + b_S)
    
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    kcal_per_g = pm.Normal('kcal_per_g', mu = mu, sd=sigma, observed = d['kcal.per.g'])
    trace_5_16 = pm.sample(1000, tune=1000)


# In[71]:


varnames = ['a', 'b_NWM', 'b_OWM', 'b_S', 'sigma']
az.summary(trace_5_16, varnames, credible_interval=.89).round(3)


# #### Code 5.52

# In[72]:


varnames = ['mu_ape', 'mu_NWM', 'b_OWM', 'b_S']
az.summary(trace_5_16, varnames, credible_interval=.89).round(3)[['mean', 'sd', 'hpd_5.5%', 'hpd_94.5%']]


# #### Code 5.53

# In[73]:


diff_NMW_OWM = trace_5_16['mu_NWM'] - trace_5_16['mu_OWM']
np.percentile(diff_NMW_OWM, 2.5), np.percentile(diff_NMW_OWM, 50), np.percentile(diff_NMW_OWM, 97.5)


# #### Code 5.54

# In[74]:


z = pd.Categorical(d['clade'])
d['clade_id'] = z.codes


# #### Code 5.55

# In[75]:


with pm.Model() as m5_16_alt:
    a = pm.Normal('a',mu = 0.6, sd=10, shape=len(d['clade_id'].unique()))
    sigma = pm.Uniform('sigma', lower= 0 , upper= 10)
    kcal_per_g = pm.Normal('kcal_per_g', mu = a[d['clade_id'].values], sd=sigma, observed = d['kcal.per.g'])
    trace_5_16_alt = pm.sample(1000, tune=1000) 


# In[76]:


varnames = ['a', 'sigma']
az.summary(trace_5_16_alt, varnames, credible_interval=.89)


# The following cells (5.56-5.61) correspond to example code for the use of R's function: lm. Therefore they have no output.

# #### Code 5.62

# In[77]:


data  = pd.read_csv('Data/cars.csv', sep=',') 
pm.GLM.from_formula('dist ~ speed', data=data)


# In[78]:


import platform
import sys

import IPython
import matplotlib
import scipy

print("""This notebook was created using:\nPython {}\nIPython {}\nPyMC {}\nArviZ {}\nNumPy {}\nSciPy {}\nMatplotlib {}\n""".format(sys.version[:5], IPython.__version__, pm.__version__, az.__version__, np.__version__, scipy.__version__, matplotlib.__version__))


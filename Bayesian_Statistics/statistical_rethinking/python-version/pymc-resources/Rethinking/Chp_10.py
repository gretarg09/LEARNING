#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import theano.tensor as tt

from scipy import stats
from scipy.special import expit as logistic

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use('arviz-darkgrid')
warnings.simplefilter(action="ignore", category=FutureWarning)


# #### Code 10.1

# In[3]:


d = pd.read_csv('Data/chimpanzees.csv', sep=";")
# we change "actor" to zero-index
d.actor = d.actor - 1


# #### Code 10.2

# In[4]:


with pm.Model() as model_10_1:
    a = pm.Normal('a', 0, 10)
    bp = pm.Normal('bp', 0, 10)
    p = pm.math.invlogit(a)    
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_10_1 = pm.sample(1000, tune=1000)


# In[5]:


df_10_1 = az.summary(trace_10_1, credible_interval=.89, round_to=2)
df_10_1


# #### Code 10.3

# In[6]:


logistic(df_10_1.iloc[:,2:4]).round(5)


# #### Code 10.4

# In[7]:


with pm.Model() as model_10_2:
    a = pm.Normal('a', 0, 10)
    bp = pm.Normal('bp', 0, 10)
    p = pm.math.invlogit(a + bp * d.prosoc_left)
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_10_2 = pm.sample(1000, tune=1000)

with pm.Model() as model_10_3:
    a = pm.Normal('a', 0, 10)
    bp = pm.Normal('bp', 0, 10)
    bpC = pm.Normal('bpC', 0, 10)
    p = pm.math.invlogit(a + (bp + bpC * d.condition) * d.prosoc_left)
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_10_3 = pm.sample(1000, tune=1000)


# #### Code 10.5

# In[8]:


comp_df = az.compare({'m10.1' : trace_10_1,
                      'm10.2' : trace_10_2,
                      'm10.3' : trace_10_3})

comp_df


# In[9]:


az.plot_compare(comp_df);


# #### Code 10.6

# In[10]:


az.summary(trace_10_3, credible_interval=.89, round_to=2)


# #### Code 10.7

# In[11]:


np.exp(0.61)


# #### Code 10.8

# In[12]:


logistic(4)


# #### Code 10.9

# In[13]:


logistic(4 + 0.61)


# #### Code 10.10 and 10.11

# In[14]:


d_pred = pd.DataFrame({'prosoc_left' : [0, 1, 0, 1], 'condition' : [0, 0, 1, 1]})
traces = [trace_10_1, trace_10_2, trace_10_3]
models = [model_10_1, model_10_2, model_10_3]


chimp_ensemble = pm.sample_ppc_w(traces=traces, models=models, samples=1000, weights=comp_df.weight.sort_index(ascending=True))


# In[15]:


rt = chimp_ensemble['pulled_left']
pred_mean = np.zeros((1000, 4))
cond = d.condition.unique()
prosoc_l = d.prosoc_left.unique()

for i in range(len(rt)):
    tmp = []
    if rt[i].size < 2:
        continue
    for cp in cond:
        for pl in prosoc_l:
            tmp.append(np.mean(rt[i][(d.prosoc_left==pl) & (d.chose_prosoc==cp)]))
    pred_mean[i] = tmp
    
ticks = range(4)
mp = pred_mean.mean(0)
az.plot_hpd(ticks, pred_mean, color='k', smooth=False)
plt.plot(mp, color='k')
plt.xticks(ticks, ("0/0","1/0","0/1","1/1"))
chimps = d.groupby(['actor', 'prosoc_left', 'condition']).agg('mean')['pulled_left'].values.reshape(7, -1)
for i in range(7):
    plt.plot(chimps[i], 'C0')

plt.ylim(0, 1.1);


# #### Code 10.12 & 10.13
# This is the same as 10.6, but in the book using MCMC rather than quadratic approximation.

# #### Code 10.14

# In[16]:


with pm.Model() as model_10_4:
    a = pm.Normal('alpha', 0, 10, shape=len(d.actor.unique()))
    bp = pm.Normal('bp', 0, 10)
    bpC = pm.Normal('bpC', 0, 10)
    p = pm.math.invlogit(a[d.actor.values] + (bp + bpC * d.condition) * d.prosoc_left)
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_10_4 = pm.sample(1000, tune=1000)


# #### Code 10.15

# In[17]:


# remember we use a zero-index
d['actor'].unique()


# #### Code 10.16

# In[18]:


az.summary(trace_10_4, credible_interval=.89, round_to=2)


# #### Code 10.17

# In[19]:


post = pm.trace_to_dataframe(trace_10_4)
post.head()


# #### Code 10.18

# In[20]:


az.plot_kde(post['alpha__1']);


# #### Code 10.19

# In[21]:


rt = pm.sample_posterior_predictive(trace_10_4, 1000, model_10_4)['pulled_left']


# In[22]:


chimp = 2
pred_mean = np.zeros((1000, 4))
cond = d.condition.unique()
prosoc_l = d.prosoc_left.unique()
for i in range(len(rt)):
    tmp = []
    for cp in cond:
        for pl in prosoc_l:
            tmp.append(np.mean(rt[i][(d.prosoc_left == pl) & 
                                     (d.chose_prosoc == cp) & 
                                     (d.actor == chimp)]))
    pred_mean[i] = tmp

ticks = range(4)
mp = pred_mean.mean(0)
hpd = pm.hpd(pred_mean, alpha=0.11)
plt.fill_between(ticks, hpd[:,1], hpd[:,0], alpha=0.25, color='k')
plt.plot(mp, color='k')
plt.xticks(ticks, ("0/0","1/0","0/1","1/1"))
chimps = d[d.actor == chimp].groupby(['condition', 'prosoc_left', ]).agg('mean')['pulled_left'].values
plt.plot(chimps, 'C0')

plt.ylim(0, 1.1);


# #### Code 10.20

# In[23]:


d_aggregated = d.groupby(['actor', 'condition', 'prosoc_left',  ])['pulled_left'].sum().reset_index()
d_aggregated.head(8)


# #### Code 10.21

# In[24]:


with pm.Model() as model_10_5:
    a = pm.Normal('alpha', 0, 10)
    bp = pm.Normal('bp', 0, 10)
    bpC = pm.Normal('bpC', 0, 10)
    p = pm.math.invlogit(a + (bp + bpC * d_aggregated.condition) * d_aggregated.prosoc_left)
    pulled_left = pm.Binomial('pulled_left', 18, p, observed=d_aggregated.pulled_left)

    trace_10_5 = pm.sample(1000, tune=1000)


# In[25]:


az.summary(trace_10_5, credible_interval=.89, round_to=2)


# In[26]:


# hacky check of similarity to 10_3, within a hundreth
np.isclose(az.summary(trace_10_5), az.summary(trace_10_3), atol=0.01)


# #### Code 10.22

# In[27]:


d_ad = pd.read_csv('./Data/UCBadmit.csv', sep=';')
d_ad.head(8)


# #### Code 10.23

# In[28]:


d_ad['male'] = (d_ad['applicant.gender'] == 'male').astype(int)

with pm.Model() as model_10_6:
    a = pm.Normal('a', 0, 10)
    bm = pm.Normal('bm', 0, 10)
    p = pm.math.invlogit(a + bm * d_ad.male)
    admit = pm.Binomial('admit', p=p, n=d_ad.applications, observed=d_ad.admit)
    
    trace_10_6 = pm.sample(1000, tune=1000)
    
with pm.Model() as model_10_7:
    a = pm.Normal('a', 0, 10)
    p = pm.math.invlogit(a)
    admit = pm.Binomial('admit', p=p, n=d_ad.applications, observed=d_ad.admit)
    
    trace_10_7 = pm.sample(1000, tune=1000)


# #### Code 10.24

# In[29]:


# Something goofy here... 
# not even close to WAIC values, larger standard error

comp_df = az.compare({'m10.6' : trace_10_6,
                      'm10.7' : trace_10_7})

comp_df


# #### Code 10.25

# In[30]:


az.summary(trace_10_6, credible_interval=.89, round_to=2)


# #### Code 10.26

# In[31]:


post = pm.trace_to_dataframe(trace_10_6)
p_admit_male = logistic(post['a'] + post['bm'])
p_admit_female = logistic(post['a'])
diff_admit = p_admit_male - p_admit_female
diff_admit.describe(percentiles=[.025, .5, .975])[['2.5%', '50%', '97.5%']]


# #### Code 10.27

# In[32]:


for i in range(6):
    x = 1 + 2 * i
    y1 = d_ad.admit[x] / d_ad.applications[x]
    y2 = d_ad.admit[x+1] / d_ad.applications[x+1]
    plt.plot([x, x+1], [y1, y2], '-C0o', lw=2)
    plt.text(x + 0.25, (y1+y2)/2 + 0.05, d_ad.dept[x])
plt.ylim(0, 1);


# #### Code 10.28

# In[33]:


d_ad['dept_id'] = pd.Categorical(d_ad['dept']).codes


# In[34]:


with pm.Model() as model_10_8:
    a = pm.Normal('a', 0, 10, shape=len(d_ad['dept'].unique()))
    p = pm.math.invlogit(a[d_ad['dept_id'].values])
    admit = pm.Binomial('admit', p=p, n=d_ad['applications'], observed=d_ad['admit'])
    
    trace_10_8 = pm.sample(1000, tune=1000)

with pm.Model() as model_10_9:
    a = pm.Normal('a', 0, 10, shape=len(d_ad['dept'].unique()))
    bm = pm.Normal('bm', 0, 10)
    p = pm.math.invlogit(a[d_ad['dept_id'].values] + bm * d_ad['male'])
    admit = pm.Binomial('admit', p=p, n=d_ad['applications'], observed=d_ad['admit'])
    
    trace_10_9 = pm.sample(1000, tune=1000)


# #### Code 10.29

# In[35]:


# WAIC values still off
# Plus warning flag

comp_df = az.compare({'m10.6' : trace_10_6,
                      'm10.7' : trace_10_7,
                      'm10.8' : trace_10_8,
                      'm10.9' : trace_10_9})

comp_df


# #### Code 10.30

# In[36]:


az.summary(trace_10_9, credible_interval=.89, round_to=2)


# #### Code 10.31
# Replicated model above but with MCMC in book.

# #### Code 10.32

# In[37]:


import statsmodels.api as sm

from patsy import dmatrix

endog = d_ad.loc[:,['admit', 'reject']].values # cbind(admit,reject)

m10_7glm = sm.GLM(endog, dmatrix('~ 1', data=d_ad), 
                  family=sm.families.Binomial())
m10_6glm = sm.GLM(endog, dmatrix('~ male', data=d_ad), 
                  family=sm.families.Binomial())
m10_8glm = sm.GLM(endog, dmatrix('~ dept_id', data=d_ad), 
                  family=sm.families.Binomial())
m10_9glm = sm.GLM(endog, dmatrix('~ male + dept_id', data=d_ad), 
                  family=sm.families.Binomial())
# res = m10_7glm.fit()
# res.summary()


# #### Code 10.33

# In[38]:


import statsmodels.formula.api as smf

m10_4glm = smf.glm(formula='pulled_left ~ actor + prosoc_left*condition - condition', data=d, 
                   family=sm.families.Binomial())


# #### Code 10.34

# In[39]:


pm.GLM.from_formula('pulled_left ~ actor + prosoc_left*condition - condition', 
                    family='binomial', data=d)   


# #### Code 10.35

# In[40]:


# outcome and predictor almost perfectly associated
y = np.hstack([np.ones(10,)*0, np.ones(10,)])
x = np.hstack([np.ones(9,)*-1, np.ones(11,)])

m_bad = smf.glm(formula='y ~ x', 
                data=pd.DataFrame({'y':y, 'x':x}), 
                family=sm.families.Binomial()).fit()
m_bad.summary()


# #### Code 10.36

# In[41]:


with pm.Model() as m_good:
    ab = pm.Normal('ab', 0, 10, shape=2)
    p = pm.math.invlogit(ab[0] + ab[1]*x)
    y_ = pm.Binomial('y_', 1, p, observed=y)
    
    MAP = pm.find_MAP()
MAP


# #### Code 10.37

# In[42]:


trace = pm.sample(1000, tune=1000, model=m_good)
tracedf = pm.trace_to_dataframe(trace)
grid = (sns.PairGrid(tracedf,
                     diag_sharey=False)
           .map_diag(sns.kdeplot)
           .map_upper(plt.scatter, alpha=0.1))


# #### Code 10.38

# In[43]:


y = stats.binom.rvs(n=1000, p=1/1000, size=100000)
np.mean(y), np.var(y)


# #### Code 10.39

# In[44]:


dk = pd.read_csv('Data/Kline', sep=';')
dk


# #### Code 10.40

# In[45]:


dk.log_pop = np.log(dk.population)
dk.contact_high = (dk.contact == "high").astype(int)


# In[46]:


from theano import shared

# casting data to theano shared variable. 
# It is for out of sample prediction from model with sampled trace
log_pop = shared(dk.log_pop.values)
contact_high = shared(dk.contact_high.values)
total_tools = shared(dk.total_tools.values)


# #### Code 10.41

# In[47]:


with pm.Model() as m_10_10:
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 1, shape=3)
    lam = pm.math.exp(a + b[0] * log_pop + b[1] * contact_high + b[2] * contact_high * log_pop)
    obs = pm.Poisson('total_tools', lam, observed=total_tools)
    trace_10_10 = pm.sample(1000, tune=1000)


# #### Code 10.42

# In[48]:


summary = az.summary(trace_10_10, credible_interval=.89)[['mean', 'sd', 'hpd_5.5%', 'hpd_94.5%']]
trace_cov = pm.trace_cov(trace_10_10, model=m_10_10)
invD = (np.sqrt(np.diag(trace_cov))**-1)[:, None]
trace_corr = pd.DataFrame(invD*trace_cov*invD.T, index=summary.index, columns=summary.index)

summary.join(trace_corr).round(2)


# In[49]:


az.plot_forest(trace_10_10);


# #### Code 10.43

# In[50]:


lambda_high = np.exp(trace_10_10['a'] + trace_10_10['b'][:,1] + (trace_10_10['b'][:,0] + trace_10_10['b'][:,2]) * 8)
lambda_low = np.exp(trace_10_10['a'] + trace_10_10['b'][:,0] * 8 )


# #### Code 10.44

# In[51]:


diff = lambda_high - lambda_low
np.sum(diff > 0) / len(diff)


# #### Code 10.45

# In[52]:


with pm.Model() as m_10_11:
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 1, shape=2)
    lam = pm.math.exp(a + b[0] * log_pop + b[1] * contact_high)
    obs = pm.Poisson('total_tools', lam, observed=total_tools)
    trace_10_11 = pm.sample(1000, tune=1000)


# #### Code 10.46

# In[53]:


with pm.Model() as m_10_12:
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 1)
    lam = pm.math.exp(a + b * log_pop)
    obs = pm.Poisson('total_tools', lam, observed=total_tools)
    trace_10_12 = pm.sample(1000, tune=1000)
    
with pm.Model() as m_10_13:
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 1)
    lam = pm.math.exp(a + b * contact_high)
    obs = pm.Poisson('total_tools', lam, observed=total_tools)
    trace_10_13 = pm.sample(1000, tune=1000)


# 
# 
# #### Code 10.47

# In[54]:


with pm.Model() as m_10_14:
    a = pm.Normal('a', 0, 100)
    lam = pm.math.exp(a)
    obs = pm.Poisson('total_tools', lam, observed=total_tools)
    trace_10_14 = pm.sample(1000, tune=1000)


# In[55]:


traces = [trace_10_10, trace_10_11, trace_10_12, trace_10_13, trace_10_14]
models = [m_10_10, m_10_11, m_10_12, m_10_13, m_10_14]
model_names = ['m10.10', 'm10.11', 'm10.12', 'm10.13','m10.14']

dictionary = dict(zip(model_names, traces))

islands_compare = az.compare(dictionary)

islands_compare


# In[56]:


az.plot_compare(islands_compare);


# #### Code 10.48

# In[57]:


# set new value for out-of-sample prediction 
log_pop_seq = np.linspace(6, 13, 30)
log_pop.set_value(np.hstack([log_pop_seq, log_pop_seq]))
contact_high.set_value(np.hstack([np.repeat(0, 30), np.repeat(1, 30)]))

islands_ensemble = pm.sample_posterior_predictive_w(traces, 10000, models,
                                                    weights=islands_compare.weight.sort_index(ascending=True))


# In[58]:


_, axes = plt.subplots(1, 1, figsize=(5, 5))
index = dk.contact_high==1
axes.scatter(np.log(dk.population)[~index], dk.total_tools[~index],
             facecolors='none', edgecolors='k', lw=1)
axes.scatter(np.log(dk.population)[index], dk.total_tools[index])

mp = islands_ensemble['total_tools'][:, :30]


axes.plot(log_pop_seq, np.median(mp, axis=0), '--', color='k')
az.plot_hpd(log_pop_seq, mp, credible_interval=.5, color='k')

mp = islands_ensemble['total_tools'][:, 30:]

axes.plot(log_pop_seq, np.median(mp, axis=0))
az.plot_hpd(log_pop_seq, mp, credible_interval=.5)

axes.set_xlabel('log-population')
axes.set_ylabel('total tools')
axes.set_xlim(6.8, 12.8)
axes.set_ylim(10, 73);


# #### Code 10.49
# This is the same as 10.41, but in the book using MCMC rather than MAP.

# #### Code 10.50

# In[59]:


log_pop_c = dk.log_pop.values - dk.log_pop.values.mean()
log_pop.set_value(log_pop_c)
contact_high.set_value(dk.contact_high.values)
total_tools.set_value(dk.total_tools.values)

with pm.Model() as m_10_10c:
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 1, shape=3)
    lam = pm.math.exp(a + b[0] * log_pop + b[1] * contact_high + b[2] * contact_high * log_pop)
    obs = pm.Poisson('total_tools', lam, observed=total_tools)
    trace_10_10c = pm.sample(1000, tune=1000)


# In[60]:


az.summary(trace_10_10c, credible_interval=.89, round_to=2)


# In[61]:


for trace in [trace_10_10, trace_10_10c]:
    tracedf = pm.trace_to_dataframe(trace)
    grid = (sns.PairGrid(tracedf,
                         diag_sharey=False)
               .map_diag(sns.kdeplot)
               .map_upper(plt.scatter, alpha=0.1))


# #### Code 10.51

# In[62]:


num_days = 30
y = np.random.poisson(1.5, num_days)


# #### Code 10.52

# In[63]:


num_weeks = 4
y_new = np.random.poisson(0.5*7, num_weeks)


# #### Code 10.53

# In[64]:


y_all = np.hstack([y, y_new])
exposure = np.hstack([np.repeat(1, 30), np.repeat(7, 4)]).astype('float')
monastery = np.hstack([np.repeat(0, 30), np.repeat(1, 4)])


# #### Code 10.54

# In[65]:


log_days = np.log(exposure)
with pm.Model() as m_10_15:
    a = pm.Normal('a', 0., 100.)
    b = pm.Normal('b', 0., 1.)
    lam = pm.math.exp(log_days + a + b*monastery)
    obs = pm.Poisson('y', lam, observed=y_all)
    trace_10_15 = pm.sample(1000, tune=1000)


# #### Code 10.55

# In[66]:


trace_10_15.add_values(dict(lambda_old=np.exp(trace_10_15['a']),
                            lambda_new=np.exp(trace_10_15['a'] + trace_10_15['b'])))

az.summary(trace_10_15, var_names=['lambda_old', 'lambda_new'],
           credible_interval=.89, round_to=2)


# #### Code 10.56

# In[2]:


# simulate career choices among 500 individuals
N = 500                 # number of individuals
income = np.arange(3) + 1 # expected income of each career
score = 0.5 * income      # scores for each career, based on income

# next line converts scores to probabilities
def softmax(w):
    e = np.exp(w)
    return e/np.sum(e, axis=0)
p = softmax(score)

# now simulate choice
# outcome career holds event type values, not counts
career = np.random.multinomial(1, p, size=N)
career = np.where(career==1)[1]
career[:11]


# #### Code 10.57

# In[3]:


with pm.Model() as m_10_16:
    b = pm.Normal('b', 0., 5.)
    s2 = b*2
    s3 = b*3
    
    p_ = pm.math.stack([0, s2, s3])
    obs = pm.Categorical('career', p=tt.nnet.softmax(p_), observed=career)
    
    trace_10_16 = pm.sample(1000, tune=2000, cores=2)
    
az.summary(trace_10_16, credible_interval=.89, round_to=2)


# #### Code 10.58

# In[4]:


N = 100

# simulate family incomes for each individual
family_income = np.random.rand(N)

# assign a unique coefficient for each type of event
b = np.arange(3)-1

p = softmax(score[:, None] + np.outer(b, family_income)).T

career = np.asarray([np.random.multinomial(1, pp) for pp in p])
career = np.where(career==1)[1]
career


# In[5]:


with pm.Model() as m_10_17:
    a23 = pm.Normal('a23', 0., 5., shape=2)
    b23 = pm.Normal('b23', 0., 5., shape=2)
    
    s2 = a23[0] + b23[0]*family_income
    s3 = a23[1] + b23[1]*family_income
    
    p_ = pm.math.stack([np.zeros(N), s2, s3]).T
    obs = pm.Categorical('career', p=tt.nnet.softmax(p_), observed=career)
    
    trace_10_17 = pm.sample(1000, tune=2000, cores=2)
    
az.summary(trace_10_17, credible_interval=.89, round_to=2)


# #### Code 10.59

# In[71]:


d_ad = pd.read_csv('./Data/UCBadmit.csv', sep=';')


# #### Code 10.60

# In[72]:


# binomial model of overall admission probability
with pm.Model() as m_binom:
    a = pm.Normal('a', 0, 100)
    p = pm.math.invlogit(a)
    admit = pm.Binomial('admit', p=p, n=d_ad.applications, observed=d_ad.admit)
    trace_binom = pm.sample(1000, tune=1000)
    
# Poisson model of overall admission rate and rejection rate
with pm.Model() as m_pois:
    a = pm.Normal('a', 0, 100, shape=2)
    lam = pm.math.exp(a)
    admit = pm.Poisson('admit', lam[0], observed=d_ad.admit)
    rej = pm.Poisson('rej', lam[1], observed=d_ad.reject)
    trace_pois = pm.sample(1000, tune=1000)


# #### Code 10.61

# In[73]:


m_binom = pm.summary(trace_binom, alpha=.11).round(2)
logistic(m_binom['mean'])


# #### Code 10.62

# In[74]:


m_pois = pm.summary(trace_pois, alpha=.11).round(2)
m_pois['mean'][0]
np.exp(m_pois['mean'][0])/(np.exp(m_pois['mean'][0])+np.exp(m_pois['mean'][1]))


# #### Code 10.63

# In[75]:


# simulate
N = 100
x = np.random.rand(N)
y = np.random.geometric(logistic(-1 + 2*x), size=N)

with pm.Model() as m_10_18:
    a = pm.Normal('a', 0, 10)
    b = pm.Normal('b', 0, 1)
    p = pm.math.invlogit(a + b*x)
    obs = pm.Geometric('y', p=p, observed=y)
    trace_10_18 = pm.sample(1000, tune=1000)
az.summary(trace_10_18, credible_interval=.89, round_to=2)


# In[76]:


import platform
import sys

import IPython
import matplotlib
import scipy

print("""This notebook was created using:\nPython {}\nIPython {}\nPyMC {}\nArviZ {}\nNumPy {}\nSciPy {}\nMatplotlib {}\n""".format(sys.version[:5], IPython.__version__, pm.__version__, az.__version__, np.__version__, scipy.__version__, matplotlib.__version__))


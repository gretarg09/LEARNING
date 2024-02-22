#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use('arviz-darkgrid')


# #### Code 9.1

# In[3]:


d = {'A':[0, 0, 10, 0, 0], 
     'B':[0, 1, 8, 1, 0], 
     'C':[0, 2, 6, 2, 0], 
     'D':[1, 2, 4, 2, 1], 
     'E':[2, 2, 2, 2, 2]}
p = pd.DataFrame(data=d)


# #### Code 9.2

# In[4]:


p_norm = p/p.sum(0)


# #### Code 9.3

# In[5]:


def entropy(x):
    y = []
    for i in x:
        if i == 0:
            y.append(0)
        else: 
            y.append(i*np.log(i))
    h = -sum(y)
    return h
H = p_norm.apply(entropy, axis=0)
H


# #### Code 9.4

# In[6]:


ways = [1, 90, 1260, 37800, 113400]
logwayspp = np.log(ways)/10
plt.plot(logwayspp, H, 'o')
plt.plot([0.0, max(logwayspp)], [0.0, max(H)], '--k')
plt.ylabel('entropy', fontsize=14)
plt.xlabel('log(ways) per pebble');


# #### Code 9.5

# In[7]:


# Build list of the candidate distributions.
p = [[1/4, 1/4, 1/4, 1/4],
     [2/6, 1/6, 1/6, 2/6],
     [1/6, 2/6, 2/6, 1/6],
     [1/8, 4/8, 2/8, 1/8]]

# Compute expected value of each. The sum of the multiplied entries is just a dot product.
p_ev = [np.dot(i, [0, 1, 1, 2]) for i in p]
p_ev


# #### Code 9.6

# In[8]:


# Compute entropy of each distribution
p_ent = [entropy(i) for i in p]
p_ent


# #### Code 9.7

# In[9]:


p = 0.7
A = [(1-p)**2, p*(1-p), (1-p)*p, p**2]
A


# #### Code 9.8

# In[10]:


-np.sum(A*np.log(A))


# #### Code 9.9

# In[11]:


def sim_p(G=1.4):
    x123 = np.random.uniform(size=3)
    x4 = (G * np.sum(x123) - x123[1] - x123[2]) / (2 - G)
    x1234 = np.concatenate((x123, [x4]))
    z = np.sum(x1234)
    p = x1234 / z
    return - np.sum(p * np.log(p)), p


# #### Code 9.10

# In[12]:


H = []
p = np.zeros((10**5, 4))
for rep in range(10**5):
    h, p_ = sim_p()
    H.append(h)
    p[rep] = p_


# In[13]:


az.plot_kde(H)
plt.xlabel('Entropy')
plt.ylabel('Density');


# #### Code 9.12

# In[14]:


np.max(H)


# #### Code 9.13

# In[15]:


p[np.argmax(H)]


# In[16]:


import platform
import sys

import IPython
import matplotlib
import scipy

print("""This notebook was created using:\nPython {}\nIPython {}\nArviZ {}\nNumPy {}\nMatplotlib {}\n""".format(sys.version[:5], IPython.__version__, az.__version__, np.__version__, matplotlib.__version__))

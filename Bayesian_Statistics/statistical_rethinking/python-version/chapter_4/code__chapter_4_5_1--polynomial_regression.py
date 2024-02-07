import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


d = pd.read_csv("../Data/Howell1.csv", sep=";", header=0)
d2 = d[d.age >= 18]



# _______________ 4.5.1 POLYNOMIAL REGRESSION _______________

# #### Code 4.64
# We have already loaded this dataset, check code 4.7 and 4.8.
d.head()

plt.figure(figsize=(8, 4))
plt.scatter(d.weight, d.height)
plt.ylabel("height")
plt.xlabel("weight")
plt.savefig('f__4_64.png')
plt.close()

# #### Code 4.65

d["weight_std"] = (d.weight - d.weight.mean()) / d.weight.std()
d["weight_std2"] = d.weight_std**2

with pm.Model() as m_4_5:
    a = pm.Normal("a", mu=178, sigma=100)
    b1 = pm.Lognormal("b1", mu=0, sigma=1)
    b2 = pm.Normal("b2", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = pm.Deterministic("mu", a + b1 * d.weight_std.values + b2 * d.weight_std2.values)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=d.height.values)
    trace_4_5 = pm.sample(1000, tune=1000)


varnames = ["~mu"]
az.plot_trace(trace_4_5, varnames);


# #### Code 4.66
print('code 4.66')
print(az.summary(trace_4_5, varnames, kind="stats", round_to=2))

# #### Code 4.67
mu_pred = trace_4_5.posterior["mu"]
trace_4_5_thinned = trace_4_5.sel(draw=slice(None, None, 5))
with m_4_5:
    height_pred = pm.sample_posterior_predictive(trace_4_5_thinned)

# #### Code 4.68
plt.figure(figsize=(8, 4))
ax = az.plot_hdi(d.weight_std, mu_pred, hdi_prob=0.89)
az.plot_hdi(d.weight_std, height_pred.posterior_predictive["height"], ax=ax, hdi_prob=0.89)
plt.scatter(d.weight_std, d.height, c="C0", alpha=0.3)
plt.savefig('f__4_68.png')
plt.close()


# #### Code 4.69
# We will stack the weights to get a 2D array, this simplifies writing a model. Now we can compute the dot product between beta and the 2D-array
weight_m = np.vstack((d.weight_std, d.weight_std**2, d.weight_std**3))
weight_m

with pm.Model() as m_4_6:
    a = pm.Normal("a", mu=178, sigma=100)
    b = pm.Normal("b", mu=0, sigma=10, shape=3)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = pm.Deterministic("mu", a + pm.math.dot(b, weight_m))
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=d.height.values)
    trace_4_6 = pm.sample(1000, tune=1000)


# #### Code 4.70 and 4.71
mu_pred = trace_4_6.posterior["mu"]
trace_4_6_thin = trace_4_6.sel(draw=slice(None, None, 5))
with m_4_6:
    height_pred = pm.sample_posterior_predictive(trace_4_6_thin)

plt.figure(figsize=(8, 4))
ax = az.plot_hdi(d.weight_std, mu_pred, hdi_prob=0.89)
az.plot_hdi(d.weight_std, height_pred.posterior_predictive["height"], ax=ax, hdi_prob=0.89)
plt.scatter(d.weight_std, d.height, c="C0", alpha=0.3)

# convert x-axis back to original scale
at = np.arange(-2, 3)
labels = np.round(at * d.weight.std() + d.weight.mean(), 1)
plt.xticks(at, labels);
plt.savefig('f__4_71.png')
plt.close()


# _______________ 4.5.2 SPLINES _______________

'''
In statistics, a spline is a smooth function built out of smaller, component functions.

B-splines build up wiggly functions from simpler less-wiggly components. Those components are
called basis functions. The basis functions are not wiggly at all, they are just step functions.

The short explanation for B-splines is that they divided the full range of some predictor variable, like year, into parts.
Tehn tehy assign a parameter to each part. These parameters are gradually turned on and off in a way that makes 
their sum into a fancy, wiggly curve.
'''

# #### Code 4.72
d = pd.read_csv("../Data/cherry_blossoms.csv")
# nans are not treated as in the book
print('code 4.72')
print(az.summary(d.dropna().to_dict(orient="list"), kind="stats"))

plt.figure(figsize=(8, 4))
fig, ax = plt.subplots(figsize=(15, 5))
plt.scatter(d.year, d.doy)
plt.xlabel("year")
plt.ylabel("day of first blossom")
plt.savefig('f__4_72.png')
plt.close()

# #### Code 4.73
d2 = d.dropna(subset=["doy"])
num_knots = 15
knot_list = np.quantile(d2.year, np.linspace(0, 1, num_knots))


# #### Code 4.74
# Here we will use patsy as a simple way of building the b-spline matrix. For more detail please read https://patsy.readthedocs.io/en/latest/spline-regression.html
from patsy import dmatrix

B = dmatrix(
    "bs(year, knots=knots, degree=3, include_intercept=True)-1",
    {"year": d2.year.values, "knots": knot_list[1:-1]},
)


# #### Code 4.75
_, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(B.shape[1]):
    ax.plot(d2.year, (B[:, i]), color="C0")
ax.set_xlabel("year")
ax.set_ylabel("basis");
plt.savefig('f__4_75.png')
plt.close()


# #### Code 4.76
# Note: if the model gets stalled instead of sampling try replacing `mu = pm.Deterministic("mu", a + pm.math.dot(B.base, w.T))` with `mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))`

with pm.Model() as m4_7:
    a = pm.Normal("a", 100, 10)
    w = pm.Normal("w", mu=0, sigma=10, shape=B.shape[1])
    # mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
    mu = pm.Deterministic("mu", a + pm.math.dot(B.base, w.T))
    sigma = pm.Exponential("sigma", 1)
    D = pm.Normal("D", mu, sigma, observed=d2.doy.values)
    trace_m4_7 = pm.sample(1000)


# #### Code 4.77
_, ax = plt.subplots(1, 1, figsize=(8, 4))
wp = trace_m4_7.posterior.w.mean(dim=["chain", "draw"])
for i in range(17):
    ax.plot(d2.year, (wp[i].item(0) * B[:, i]), color="C0")
ax.set_xlim(812, 2015)
ax.set_ylim(-6, 6);
plt.savefig('f__4_77.png')
plt.close()


# #### Code 4.78
ax = az.plot_hdi(d2.year, trace_m4_7.posterior["mu"], color="k")
ax.plot(d2.year, d2.doy, "o", alpha=0.3)
fig = plt.gcf()
fig.set_size_inches(8, 4)
ax.set_xlabel("year")
ax.set_ylabel("days in year")
plt.savefig('f__4_78.png')
plt.close()
# get_ipython().run_line_magic('load_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '-n -u -v -iv -w -p aesara,aeppl,xarray')

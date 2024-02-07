import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


d = pd.read_csv("../Data/Howell1.csv", sep=";", header=0)
d2 = d[d.age >= 18]

# _______________ 4.5.2 SPLINES _______________

'''
In statistics, a spline is a smooth function built out of smaller, component functions.

B-splines build up wiggly functions from simpler less-wiggly components. Those components are
called basis functions. The basis functions are not wiggly at all, they are just step functions.

The short explanation for B-splines is that they divided the full range of some predictor variable, like year, into parts.
Then they assign a parameter to each part. These parameters are gradually turned on and off in a way that makes 
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
    "bs(year, knots=knots, degree=2, include_intercept=True) - 1",
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

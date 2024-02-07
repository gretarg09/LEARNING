import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix


plt.figure(figsize=(8, 4))
plt.title("B-spline basis example (degree=3)");
x = np.linspace(0., 1., 100)
y = dmatrix("bs(x, df=6, degree=3, include_intercept=True) - 1", {"x": x})

# Define some coefficients
b = np.array([1.3, 0.6, 0.9, 0.4, 1.6, 0.7])

# Plot B-spline basis functions (colored curves) each multiplied by its coefficient.
plt.plot(x, y*b);
plt.plot(x, np.dot(y, b), color='k', linewidth=3);
plt.show()
plt.close()

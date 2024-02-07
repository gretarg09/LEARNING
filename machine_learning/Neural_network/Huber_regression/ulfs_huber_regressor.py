import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Series
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

class HuberRegressor(BaseEstimator, RegressorMixin):
    """Huber regression
    
    This code was implemented by Ulf

    Huber regression is a robust regression method that is less sensitive to
    outliers than least squares regression. It works exactly like ordinary
    linear regression, however with a Huber loss function instead of the
    squared error loss function. The Huber loss function is a combination of
    the squared error loss function and the absolute error loss function. It
    uses squared error for small errors and absolute error for large errors.

    Args:
        epsilon : Huber loss hyperparameter. Defaults to 1.35.
    """

    def __init__(self, epsilon=1.35):
        self.model = None
        self.epsilon = epsilon

    def __call__(self, X):
        return self.predict(X)

    def huber_loss(self, params, X, y, epsilon=1.35):
        intercept, coeffs = params[0], params[1:]
        diff = y - (intercept + np.dot(X, coeffs))
        loss = np.where(
            np.abs(diff) < epsilon,
            0.5 * diff**2,
            epsilon * np.abs(diff) - 0.5 * epsilon**2,
        )
        return np.sum(loss)

    def fit( self, X, y) -> "HuberRegressor":
        """Fit the Huber regression model using `sklearn.linear_model.HuberRegressor`.

        Args:
            X: array of features
            y: array of target variable

        Returns
            self
        """
        initial_params = np.zeros(X.shape[1] + 1) / X.shape[1]

        coef_constraints = {
            'type': 'ineq',
            'fun': lambda params: params,
        }

        result = minimize(
            self.huber_loss, initial_params, args=(X, y), constraints=coef_constraints
        )

        self.intercept_ = result.x[0]
        self.coef_ = pd.Series(result.x[1:], index=X.columns)

        return self

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_

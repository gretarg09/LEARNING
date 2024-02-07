from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.signal import convolve2d
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution
from joblib import dump, load

import numpy as np


'''
https://towardsdatascience.com/an-upgraded-marketing-mix-modeling-in-python-5ebb3bddc1b6

Adding adstock to our marketing mix model 

    Saturation Effect:
        We want to create a transformation (=mathematical function) with the following properties:

        1. If the spendings are 0, the saturated spendings are also 0.
        2. The transformation is monotonically increasing, i.e.,
           the higher the input spendings, the higher the saturated output spendings.
        3. The saturated values do not grow to infinity. Instead, they are upper bounded
           by some number, let us say 1.

    Carryover Effect:
        We now want that spendings in one week to get partially carried over to the 
        following weeks in an exponential fashion. This means: In week 1 there is a spend of 16.
        Then we carry over 50%, meaning

        Example:
            (16, 0, 0, 0, 0, 4, 8, 0, 0, 0) -> (16, 8, 4, 0, 0, 4, 10, 5, 2, 0).


    We will use different saturation and carryover effects for each channel.
    This makes sense because a TV ad usually sticks longer in your head tahn a banner 
    you see online, for example.
'''


class ExponentialSaturation(BaseEstimator, TransformerMixin):
    '''
    Take a look at the article 
    '''
    def __init__(self, a=1.):
        self.a = a
        
    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True) # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False) # from BaseEstimator
        return 1 - np.exp(-self.a*X)




class ExponentialCarryover(BaseEstimator, TransformerMixin):
    def __init__(self, strength=0.5, length=1):
        self.strength = strength
        self.length = length

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (
            self.strength ** np.arange(self.length + 1)
        ).reshape(-1, 1)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_)
        if self.length > 0:
            convolution = convolution[: -self.length]
        return convolution

pipelines = {
        'tv':  Pipeline([('carryover', ExponentialCarryover()),
                         ('saturation', ExponentialSaturation())]),
        'radio': Pipeline([('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())]),
        'banners': Pipeline([('carryover', ExponentialCarryover()),
                             ('saturation', ExponentialSaturation())])
    }

adstock = ColumnTransformer(
    [
         ('tv_pipe', pipelines['tv'], ['TV']),
         ('radio_pipe', pipelines['radio'] , ['Radio']),
         ('banners_pipe',  pipelines['banners'], ['Banners']),
    ],
    remainder='passthrough'
)

model = Pipeline([
                  ('adstock', adstock),
                  ('regression', LinearRegression())
])



# Let's load the data again and fit a simple model
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

data = pd.read_csv(
    'https://raw.githubusercontent.com/Garve/datasets/4576d323bf2b66c906d5130d686245ad205505cf/mmm.csv',
    parse_dates=['Date'],
    index_col='Date'
)
X = data.drop(columns=['Sales'])
y = data['Sales']

model.fit(X, y)

print(cross_val_score(model, X, y, cv=TimeSeriesSplit()).mean())

# Output: ~0.55



# try to load up the model before running


try: 
    tuned_model = load('marketing_mix_model__upgraded.joblib')
    print('found model')
except:
    print('Model not found. Running hyperparameter tuning...')
    tuned_model = None
    pass


if not tuned_model:

    # HYPERPARAMETER TUNING

    tuned_model = OptunaSearchCV(
        estimator=model,
        param_distributions={
            'adstock__tv_pipe__carryover__strength': FloatDistribution(0, 1),
            'adstock__tv_pipe__carryover__length': IntDistribution(0, 6),
            'adstock__tv_pipe__saturation__a': FloatDistribution(0, 0.01),
            'adstock__radio_pipe__carryover__strength': FloatDistribution(0, 1),
            'adstock__radio_pipe__carryover__length': IntDistribution(0, 6),
            'adstock__radio_pipe__saturation__a': FloatDistribution(0, 0.01),
            'adstock__banners_pipe__carryover__strength': FloatDistribution(0, 1),
            'adstock__banners_pipe__carryover__length': IntDistribution(0, 6),
            'adstock__banners_pipe__saturation__a': FloatDistribution(0, 0.01),
        },
        n_trials=1000,
        cv=TimeSeriesSplit(),
        random_state=0
    )

    # Run the optimized model
    print(cross_val_score(tuned_model, X, y, cv=TimeSeriesSplit()))


    print(tuned_model.best_params_)
    print(tuned_model.best_estimator_.named_steps['regression'].coef_)
    print(tuned_model.best_estimator_.named_steps['regression'].intercept_)

    dump(tuned_model, 'marketing_mix_model__upgraded.joblib') 

# Output:
# Hyperparameters = {
# 'adstock__tv_pipe__carryover__strength': 0.5248878517291329
# 'adstock__tv_pipe__carryover__length': 4
# 'adstock__tv_pipe__saturation__a': 1.4649722346562529e-05
# 'adstock__radio_pipe__carryover__strength': 0.45523455448406197
# 'adstock__radio_pipe__carryover__length': 0
# 'adstock__radio_pipe__saturation__a': 0.0001974038926379962
# 'adstock__banners_pipe__carryover__strength': 0.3340342963936898
# 'adstock__banners_pipe__carryover__length': 0
# 'adstock__banners_pipe__saturation__a': 0.007256873558015173
# }
#
# Coefficients = [27926.6810003   4114.46117033  2537.18883927]
# Intercept = 5348.966158957056

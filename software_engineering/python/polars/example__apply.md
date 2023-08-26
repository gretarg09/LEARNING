##  Combining apply, user defined function and pl.when 

```
import pandas as pd
import numpy as np
import polars as pl

df = pd.DataFrame({
     'target_type': ['revenue', 'conversions', 'revenue'],
     'opt_fit_total_new_revenue': [105, np.nan, 85],
     'new_revenue': [90, np.nan, 95],
     'opt_fit_total_new_conversions': [np.nan, 155, np.nan],
     'new_conversions': [np.nan, 240, np.nan],
     'spend_change_suggestion': [10, 10, -10]
})

columns = ['opt_fit_total_new_revenue',
           'new_revenue',
           'opt_fit_total_new_conversions',
           'new_conversions',
           'spend_change_suggestion']

def rate_of_change(change_col, actual_col, spend_change_suggestion):
    if change_col == None or actual_col == None or spend_change_suggestion == None:
        return None
    return (change_col - actual_col) / spend_change_suggestion

revenue = pl.struct(['opt_fit_total_new_revenue', 'new_revenue', 'spend_change_suggestion']).apply(lambda x: rate_of_change(x['opt_fit_total_new_revenue'],
                                                                                                                            x['new_revenue'],
                                                                                                                            x['spend_change_suggestion'])).alias('secant')
conversion = pl.struct(['opt_fit_total_new_conversions', 'new_conversions', 'spend_change_suggestion']).apply(lambda x: rate_of_change(x['opt_fit_total_new_conversions'],
                                                                                                                                       x['new_conversions'],
                                                                                                                                       x['spend_change_suggestion'])).alias('secant')

df =  df.with_columns(pl.when(pl.col('target_type') == 'revenue').then(revenue).otherwise(conversion))
```

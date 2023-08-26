## User defined function on the whole dataframe

This is an example of how the map operator can be used to execute a user defined function on the whole dataframe.
Here I am using it to create a new column, the column is calculated based on existing subset of columns. 
```
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'target_type': ['revenue', 'conversions', 'revenue'],
    'opt_fit_total_new_revenue': [105, np.nan, 85],
    'new_revenue': [90, np.nan, 95],
    'opt_fit_total_new_conversions': [np.nan, 155, np.nan],
    'new_conversions': [np.nan, 140, np.nan],
    'spend_change_suggestion': [10, 10, -10]
 })

def rate_of_change(row, change_col, actual_col, spend_col):
    if pd.isna(row[change_col]) or pd.isna(row[actual_col]) or pd.isna(row[spend_col]):
        return np.nan
    return (row[change_col] - row[actual_col]) / row[spend_col]


def compute_secant(row):
    if row['target_type'] == 'revenue':
        return rate_of_change(row, 'opt_fit_total_new_revenue', 'new_revenue', 'spend_change_suggestion')
    else:
        return rate_of_change(row, 'opt_fit_total_new_conversions', 'new_conversions', 'spend_change_suggestion')

df['secant'] = df.apply(compute_secant, axis=2)
```

# groupby by a list

[pandas groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)
In pandas it is possible to pass in a list as the by parameter in the group by function. As stated in the documentation
`If a list or ndarray of length equal to the selected axis is passed, the values are used as-is to determine the groups`
This means that you can label each row of the dataframe yourself, and the group by those labels, by providing a list
equal to the length of the dataframe. 

Lets illustrate this with an example:

```
import pandas as pd
import datetime

# Create a list of datetime.date objects
dates = [datetime.date(2023, 9, 15),
         datetime.date(2023, 9, 16),
         datetime.date(2023, 9, 17),
         datetime.date(2023, 9, 18),
         datetime.date(2023, 9, 19),
         datetime.date(2023, 9, 20),
         datetime.date(2023, 9, 21),
         datetime.date(2023, 9, 22),
         datetime.date(2023, 9, 23),
         datetime.date(2023, 9, 24)
]

 # Create some sample data
 data = {'Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
         'Column2': [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000]}

# Create a DataFrame with datetime.date objects as the index
df = pd.DataFrame(data, index=dates)

weekdays = [d.weekday() for d in df.index]
weekdays_1 = [4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
weekdays_2 = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]

#df.groupby(weekdays).mean()
print(df)
print('')
print(df.groupby(weekdays_1).mean())
print('')
print(df.groupby(weekdays_2).mean())
```


# References 
[Everything you need to know about loc and iloc](https://towardsdatascience.com/everything-you-need-to-know-about-loc-and-iloc-of-pandas-79b386cac776)

* `loc`: select by labels of rows and columns.
* `iloc`: select by positions of rows and columns.



# Use loc to create a bigger dataframe

 loc can be used to get data for a list that is bigger than the length of the dataframe. The list of course needs to only contain 
 labels that are within the index of the dataframe. Let's showcase this with an example:

 ```
import pandas as pd
import datetime

# Create a list of datetime.date objects
i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

# Create some sample data
data = {'Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Column2': [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000]}

# Create a DataFrame with datetime.date objects as the index

df = pd.DataFrame(data, index=i)

print(df)
print()
print(df.loc[[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 ]])
 ```

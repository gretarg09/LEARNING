from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import pandas as pd
from matplotlib import pyplot as plt


'''
Simple Marketing Mix Modeling
-----------------------------
https://towardsdatascience.com/introduction-to-marketing-mix-modeling-in-python-d0dd81f4e794

Sales = f(tv) + g(radio) + h(banners) + base
'''

data = pd.read_csv(
    'https://raw.githubusercontent.com/Garve/datasets/4576d323bf2b66c906d5130d686245ad205505cf/mmm.csv',
    parse_dates=['Date'],
    index_col='Date'
)

X = data.drop(columns=['Sales'])
y = data['Sales']

lr = LinearRegression()

print(cross_val_score(lr, X, y, cv=TimeSeriesSplit()))


lr.fit(X, y) # refit the model with the complete dataset

print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)


weights = pd.Series(
    lr.coef_,
    index=X.columns
)

base = lr.intercept_

unadj_contributions = X.mul(weights).assign(Base=base)

adj_contributions = (unadj_contributions
                     .div(unadj_contributions.sum(axis=1), axis=0)
                     .mul(y, axis=0)
                    ) # contains all contributions for each day

ax = (adj_contributions[['Base', 'Banners', 'Radio', 'TV']]
      .plot.area(
          figsize=(16, 10),
          linewidth=1,
          title='Predicted Sales and Breakdown',
          ylabel='Sales',
          xlabel='Date')
     )

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1], labels[::-1],
    title='Channels', loc="center left",
    bbox_to_anchor=(1.01, 0.5)
)

'''
    channel_ROI = sales_from_channel / spendings_on_channel
'''

def channel_roi(channel):
    sales_from_channel = adj_contributions[channel].sum()
    spendings_on_channel = data[channel].sum()
    return sales_from_channel / spendings_on_channel

print(f'TV ROI: {channel_roi("TV")}')
print(f'Banners ROI: {channel_roi("Banners")}')
print(f'Radio ROI: {channel_roi("Radio")}')


plt.show()

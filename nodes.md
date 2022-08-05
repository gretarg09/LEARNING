# Pipeline visualization

In order to visualize the data pipeline run the following:
```
kedro viz
```

**[Problem]**
I had a bit of a problem when I was running this again and again. The error that I got was that the port 4141 (the default port for the visualization) was busy (or taken). In order to solve this I executed the following command:

```
user -k 4141/tcp
```

# Loading data

The kedro framework ships with an interactive python shell what can be used to load data. The shell can be activated by executing the following command:
```
kedro ipython
```

Within the interactive shell the dataset can be loaded according to the data catalogs key name. For example if I have the following dataset registered within the data catalogs
```
reviews:
  type: pandas.CSVDataSet
  filepath: data/01_raw/reviews.csv
```

I can load up this data by simply running:
```
catalog.load('reviews').head()
```

This is quite handy :).


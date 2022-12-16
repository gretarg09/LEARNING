# Resources

* [What is dbt](https://docs.getdbt.com/docs/introduction)
* [End to End DBT project in Google Cloud Platform](https://blog.devgenius.io/end-to-end-dbt-project-in-google-cloud-platform-part-1-ea14dd11cf9e)
* [DBT at scale](https://www.astrafy.io/articles/dbt-at-scale-on-google-cloud-part-1)
* [Video series about dbt](https://www.youtube.com/playlist?list=PLy4OcwImJzBLJzLYxpxaPUmCWp8j1esvT)
* [dbt courses](https://courses.getdbt.com/courses)



## BigQuery setup

* [Setting up big query](https://docs.getdbt.com/reference/warehouse-setups/bigquery-setup)


## An idea

* use [this](https://www.entechlog.com/blog/data/how-to-configure-dbt-for-postgres/) as a template on how to setup dbt locally
using a postgresql database as data warehouse and how to host the code within a docker image on my machine. Then I should 
try to use the official tutorial from dbt to run the data pipelines locally (within the docker image). 

The datasets that I will be using is the following. I just need to save this dataset down (take a subset of the dataset if it is 
too big). 
```sql
select * from `dbt-tutorial.jaffle_shop.customers`;
select * from `dbt-tutorial.jaffle_shop.orders`;
select * from `dbt-tutorial.stripe.payment`;
```

I will call this a first dbt project.

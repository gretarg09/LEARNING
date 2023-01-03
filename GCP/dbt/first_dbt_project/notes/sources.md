# Sources
* Sources represent the raw data that is loaded into the data warehouse.
* We can reference tables in our models with an explicit table name ```(raw.jaffle_shop.customers)```.
* However, setting up Sources in dbt and referring to them with the source function enables a few important tools.
    * Multiple tables from a single source can be configured in one place.
    * Sources are easily identified as green nodes in the Lineage Graph.
    * You can use dbt source freshness to check the freshness of raw tables.

# Configuring sources
* Sources are configured in YML files in the models directory.
* The following code block configures the table ```raw.jaffle_shop.customers``` and ```raw.jaffle_shop.orders```:

![](./images/sources_image_1.png)

* View the full documentation for configuring sources on the [source properties](https://docs.getdbt.com/reference/source-properties) page of the docs.

# Source function
* The ref function is used to build dependencies between models.
* Similarly, the source function is used to build the dependency of one model to a source.
* Given the source configuration above, the snippet ```{{ source('jaffle_shop','customers') }}``` in a model file will compile to ```raw.jaffle_shop.customers```.
* The Lineage Graph will represent the sources in green.

![](./images/sources_image_2.png)

# Source freshness
* Freshness thresholds can be set in the YML file where sources are configured. For each table, the keys ```loaded_at_field``` and ```freshness``` must be configured.

![](./images/sources_image_3.png)

* A threshold can be configured for giving a warning and an error with the keys ```warn_after``` and ```error_after```.
* The freshness of sources can then be determined with the command dbt source freshness.



**[source]**
[source](https://courses.getdbt.com/courses/take/fundamentals/texts/27962334-review)

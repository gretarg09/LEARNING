# Testing

* Testing is used in software engineering to make sure that the code does what we expect it to.
* In Analytics Engineering, testing allows us to make sure that the SQL transformations we write produce a model that meets our assertions.
* In dbt, tests are written as select statements. These select statements are run against your materialized models to ensure they meet your assertions.

# Tests in dbt

* In dbt, there are two types of tests - generic tests and singular tests:
    * Generic tests are written in YAML and return the number of records that do not meet your assertions. These are run on specific columns in a model.
    * Singular tests are specific queries that you run against your models. These are run on the entire model.
* dbt ships with four built in tests: unique, not null, accepted values, relationships.
    * **Unique** tests to see if every value in a column is unique
    * **Not_null** tests to see if every value in a column is not null
    * **Accepted_values** tests to make sure every value in a column is equal to a value in a provided list
    * **Relationships** tests to ensure that every value in a column exists in a column in another model (see: [referential integrity](https://en.wikipedia.org/wiki/Referential_integrity))
* Generic tests are configured in a YAML file, whereas singular tests are stored as select statements in the tests folder.
* Tests can be run against your current project using a range of commands:
    * ```dbt test``` runs all tests in the dbt project.
    * ```dbt test --select test_type:generic```
    * ```dbt test --select test_type:singular```
    * ```dbt test --select one_specific_model```
* Read more here in [testing documentation](https://docs.getdbt.com/reference/node-selection/test-selection-examples).


# Unit testing

* https://github.com/EqualExperts/dbt-unit-testing
* https://www.equalexperts.com/blog/our-thinking/writing-unit-tests-for-dbt-with-tdd/

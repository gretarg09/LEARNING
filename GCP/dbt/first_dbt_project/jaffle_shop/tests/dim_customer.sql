{{ config( tags=['unit-test', 'no-db-dependency']) }}

{% call dbt_unit_testing.test('dim_customers', 'should sum order values to calculate customer_lifetime_value') %}
  

  {% call dbt_unit_testing.mock_ref ('stg_customers') %}

    customer_id | first_name    | last_name
    1           | 'Gretar Atli' | 'Gretarsson'

  {% endcall %}
  
  {% call dbt_unit_testing.mock_ref ('fct_orders') %}

    order_id |  customer_id|    order_date  | amount
    1001     |            1|    '2018-01-01'|    200
    1002     |            1|    '2018-01-02'|    100

  {% endcall %}
  

  {% call dbt_unit_testing.expect() %}

    customer_id|    first_name|     last_name|   first_order_date|   most_recent_order_date| number_of_orders| lifetime_value
    1          | 'Gretar Atli'|  'Gretarsson'|       '2018-01-01'|             '2018-01-02'|                2|            300

  {% endcall %}

{% endcall %}

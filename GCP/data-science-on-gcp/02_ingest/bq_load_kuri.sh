BUCKET=ds-group-gretar-dsongcp

bq load --autodetect --source_format=CSV dsongcp.flights_auto gs://${BUCKET}/flights/raw/201501.csv

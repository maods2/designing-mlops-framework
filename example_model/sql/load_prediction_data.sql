-- Prediction input query for the example model.
-- Parameters are injected at runtime from YAML optional_configs.
--
-- Available parameters:
--   {project}            - GCP project ID
--   {dataset}            - BigQuery dataset name
--   {table}              - Source table name
--   {scoring_date}       - Date to score (YYYY-MM-DD)
--   {unique_id_column}   - Column that uniquely identifies each row

SELECT
    {unique_id_column},
    f0,
    f1,
    f2,
    f3,
    f4
FROM
    `{project}.{dataset}.{table}`
WHERE
    event_date = '{scoring_date}'

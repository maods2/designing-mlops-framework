-- Training data query for the example model.
-- Parameters are injected at runtime from YAML optional_configs.
--
-- Available parameters:
--   {project}            - GCP project ID
--   {dataset}            - BigQuery dataset name
--   {table}              - Source table name
--   {start_date}         - Training window start (YYYY-MM-DD)
--   {end_date}           - Training window end   (YYYY-MM-DD)
--   {sample_fraction}    - Fraction of rows to sample (0.0-1.0)
--   {target_column}      - Name of the target/label column

SELECT
    f0,
    f1,
    f2,
    f3,
    f4,
    {target_column} AS target
FROM
    `{project}.{dataset}.{table}`
WHERE
    event_date BETWEEN '{start_date}' AND '{end_date}'
    AND {target_column} IS NOT NULL
    AND RAND() < {sample_fraction}

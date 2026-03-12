FROM apache/airflow:2.10.5-python3.11

USER airflow

COPY --chown=airflow:root requirements-airflow.txt /requirements-airflow.txt

RUN pip install --no-cache-dir -r /requirements-airflow.txt

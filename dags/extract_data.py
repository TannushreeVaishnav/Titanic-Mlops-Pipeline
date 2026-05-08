from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk.bases.hook import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy


def load_to_sql(file_path: str):
    # Get connection details from Airflow's connection named 'postgres_default'
    conn = BaseHook.get_connection('postgres_default')

    # Build SQLAlchemy engine using connection info
    engine = sqlalchemy.create_engine(
        f"postgresql+psycopg2://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema or 'postgres'}"
    )

    # Load CSV into DataFrame
    df = pd.read_csv(file_path)

    # Write DataFrame to Postgres table
    df.to_sql(name="titanic", con=engine, if_exists="replace", index=False)


with DAG(
    dag_id="extract_titanic_data",
    schedule=None, 
    start_date=datetime(2026, 5, 1),
    catchup=False,
    tags=["example", "postgres", "titanic"],
) as dag:

    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/usr/local/airflow/data/Titanic-Dataset.csv"},
    )

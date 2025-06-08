from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from models.model3 import all_in_one
from models.model4 import all_in_one_tree


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

with DAG(
    'my_task_2',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=all_in_one,
    )
    load_data_task_tree = PythonOperator(
        task_id='load_data_tree',
        python_callable=all_in_one_tree,
    )

    load_data_task >> load_data_task_tree
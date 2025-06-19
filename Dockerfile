FROM apache/airflow:2.8.3-python3.10

WORKDIR /

COPY requirements.txt /

RUN pip install --upgrade setuptools pip wheel && pip install -r /requirements.txt
RUN airflow db init
RUN airflow users create --username admin --firstname YOUR_FIRST_NAME --lastname YOUR_LAST_NAME --role Admin --email YOUR_EMAIL@example.com --password admin


ENTRYPOINT [ "airflow", "standalone" ]
FROM python:3.11

WORKDIR /app

COPY ./app.py .

COPY ./requirements.txt .

COPY ./DictItem.py .

COPY ./database.py .

COPY ./RegressionObject.py .

COPY ./conftest.py .

COPY ./test_RegressionObject.py .

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2

#RUN pip install Cython --install-option="--no-cython-compile"

RUN pip install -U scikit-learn

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 4000

#RUN [ "python3", "./test_regressionObject.py" ]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=4000"]
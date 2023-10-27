FROM python:3.10

COPY ./src /code/app
COPY ./requirements.txt /code/requirements.txt

COPY ./model/model.gzip /code/app/model.gzip

WORKDIR /code

RUN pip install -r /code/requirements.txt

CMD [ "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80" ]
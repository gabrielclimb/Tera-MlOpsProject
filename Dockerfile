FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt

COPY ./src /code/app

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]
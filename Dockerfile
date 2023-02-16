FROM python:3.8.5-alpine

COPY . /app
RUN apk update && apk upgrade && apk add --no-cache --virtual build-deps gcc python3-dev
RUN pip install -r /app/requirements.txt

ENTRYPOINT [ "python", "/app/src/train.py" ]
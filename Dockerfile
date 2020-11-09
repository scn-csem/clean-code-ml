FROM python:3.6-slim

RUN apt-get update
RUN apt install vim less -y

WORKDIR /code
COPY . /code

RUN pip install -U pip && pip install -r requirements.txt

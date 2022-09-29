FROM python:3.8-slim-buster

WORKDIR /LocationNet
COPY . /LocationNet

RUN pip install -r requirements.txt

WORKDIR src

CMD python app.py
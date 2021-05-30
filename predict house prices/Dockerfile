# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

ADD ./python_requirements.txt /
RUN pip3 install -r python_requirements.txt

ADD ./prediction.py /
ADD ./server.py /
ADD ./rf.joblib /

CMD [ "python3", "-u", "./server.py" ]

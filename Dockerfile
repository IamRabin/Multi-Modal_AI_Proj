
FROM ubuntu:18.04
FROM tensorflow/tensorflow:2.2.0

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev python3-dev

COPY . source/
RUN /bin/bash -c "cd source \
     && pip install -r requirements.txt"

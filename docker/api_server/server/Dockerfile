# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.6-buster

# Create the working directory
RUN set -ex && mkdir /app
WORKDIR /app

# Install Python dependencies
COPY ./server/requirements.txt ./requirements.txt
RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y git-lfs
RUN git lfs install


EXPOSE 8000
ENV PYTHONPATH /app

# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.6-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Create the working directory
RUN set -ex && mkdir /app
WORKDIR /app

# Install Python dependencies
COPY ./requirements.txt ./requirements.txt
RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "training/run_experiment.py"]

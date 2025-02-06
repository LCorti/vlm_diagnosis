# Use an official Python runtime as the base image
FROM python:3.10.5

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential

RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for Flask's run host
ENV FLASK_RUN_HOST=0.0.0.0

CMD gunicorn --preload --bind 0.0.0.0:8000 app:app --timeout 10800 --worker-class=gevent --workers 4

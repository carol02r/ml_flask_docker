# base image with fewer vulnerabilities
FROM python:3.11-slim

COPY requirements.txt .

# installing all dependencies
RUN pip install -r requirements.txt

# copy everyhting in directory app
COPY . .

# set environment variable for Flask
ENV FLASK_APP=app.py

# container will listen on port 4000
EXPOSE 4000

CMD ["flask", "run", "--host=0.0.0.0", "--port=4000"]

# in terminal...

## create image:
# app % docker compose up -- build

## store image on docker hub as public repo:
# app % docker compose push

## check images in local machine:
# app % docker images
## check containers in local machine:
# app % docker ps -a

## remove images by id
# app % docker rmi -f image id
## remove containers by id
# app % docker rm -f container id

## get container from my public repo (on docker hub)
## pulls into into docker desktop
# docker run -p4000:4000 carol02r/dare_study
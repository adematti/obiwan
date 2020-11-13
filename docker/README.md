# Building the Docker image

First create an account `youraccount` at <https://hub.docker.com>.

To build, go to the project root directory and run:
```bash
docker-compose start
docker-compose build
```
Or, alternatively:
```bash
docker build -f docker/Dockerfile -t obiwan .
```
To tag and push:
```bash
docker tag obiwan youraccount/obiwan:tag
docker login
docker push youraccount/obiwan:tag
```

# Building the Docker image

First create an account `youraccount` at <https://hub.docker.com>.

To build:
```
docker build -t obiwan .
```

To tag and push:
```
docker tag obiwan youraccount/obiwan:tag
docker login
docker push youraccount/obiwan:tag
```
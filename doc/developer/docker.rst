Building the Docker image
=========================

First create an account ``youraccount`` at `<https://hub.docker.com>`_.

To build, go into the root directory and run::

  docker-compose start
  docker-compose build

Or, alternatively::

   docker build -f docker/Dockerfile -t obiwan .

To tag and push::

  docker tag obiwan youraccount/obiwan:tag
  docker login
  docker push youraccount/obiwan:tag

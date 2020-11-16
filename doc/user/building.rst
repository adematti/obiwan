Building
########

The software stack is rather complicated. **Obiwan** is a wrapper around **legacypipe**, which has many dependences.
So there are two methods for installing everything, **Docker** and **desiconda**.

Obiwan with Docker
********************

**Docker** is the preferred method, as **Obiwan** can be run from a laptop and NERSC using the same **Docker** image.

The **Docker** image is available on Docker Hub: `Obiwan docker <https://hub.docker.com/r/adematti/obiwan>`_. First pull::

  docker pull adematti/obiwan:dr9.3

To run on-the-fly::

  docker run adematti/obiwan:dr9.3 ./yourscript.sh

Or in interactive mode, you can bind mount your working directory ``absolutepath``::

  docker run -v absolutepath:/homedir/ -it adematti/obiwan:dr9.3 /bin/bash

which allows you to work as usual (type ``exit`` to exit). On NERSC::

  shifterimg -v pull adematti/obiwan:dr9.3
  shifter --module=mpich-cle6 --image=adematti/obiwan:dr9.3 ./yourscript.sh

In interactive mode::

  shifter -v absolutepath:/homedir/ --image=adematti/obiwan:dr9.3 /bin/bash

Obiwan from scratch
*******************

To install **Obiwan** from scratch, the simplest way is to follow instructions in the `Travis file <https://github.com/adematti/obiwan/blob/master/.travis.yml>`_:
first go through ``apt`` installs, then ``install``.

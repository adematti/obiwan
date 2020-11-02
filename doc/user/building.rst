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

Or in interactive mode, you can bind mount your directory ``obiwanabsolutepath``::

  docker run -v obiwanabsolutepath:/src/obiwan -it adematti/obiwan:dr9.3

which allows you to work as usual (type ``exit`` to exit). On NERSC::

  shifterimg -v pull adematti/obiwan:dr9.3
  shifter --module=mpich-cle6 --image=adematti/obiwan:dr9.3 ./yourscript.sh


Obiwan with **desiconda**
*************************

In construction.

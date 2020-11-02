# Obiwan

**Obiwan** is a Monte Carlo method for adding fake galaxies and stars to images from the Legacy Survey and re-processing the modified images with the [Legacysurvey/Tractor pipeline](https://github.com/legacysurvey/legacypipe). The pipeline forward models galaxies and stars in the multi-color images by detecting sources with Signal to Noise (S/N) greater than 6 and minimizing the regularized L2 Loss function for various models for the shapes of stars and galaxies.

## Credits

Note this git repo is in construction.
All credits to Hui Kong, Kaylan Burleigh, John Moustakas.
See the [offical acknowledgements](http://legacysurvey.org/#Acknowledgements) for the Legacy Survey.
Applied to eBOSS ELG in [Obiwan on eBOSS](https://arxiv.org/abs/2007.08992).


## Documentation

Documentation (in construction) is hosted on Read the Docs, [Obiwan Docs](https://obiwandr9.readthedocs.io/).

## Installation

A Docker image is available on Docker Hub:
<https://hub.docker.com/r/adematti/obiwan>
```
docker pull adematti/obiwan:dr9.3
```
To run on-the-fly:
```
docker run adematti/obiwan:dr9.3 ./yourscript.sh
```
Or in interactive mode, you can bind mount your directory `obiwanabsolutepath`:
```
docker run -v obiwanabsolutepath:/src/obiwan -it adematti/obiwan:dr9.3
```
which allows you to work as usual (type `exit` to exit).<br>
On NERSC:
```
shifterimg -v pull adematti/obiwan:dr9.3
shifter --module=mpich-cle6 --image=adematti/obiwan:dr9.3 ./yourscript.sh
```

## License

**Obiwan** is free software licensed under a 3-clause BSD-style license. For details see the [LICENSE](https://github.com/adematti/obiwan/blob/master/LICENSE).

## Requirements

-   Python 3
-   numpy
-   scipy
-   matplotlib
-   fitsio
-   cython
-   mpi4py
-   h5py
-   pandas
-   pytest
-   astropy
-   photutils
-   astrometry.net
-   tractor
-   legacypipe
-   galsim

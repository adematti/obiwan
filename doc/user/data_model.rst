Data model
##########

In a nutshell, outputs are written in the following structure::

  test_brick
  |-- randoms
  |   |-- README.txt
  |   `-- randoms_seed_1_startid_1.fits
  |-- logs
  |   `-- 135
  |       `-- 1351p192
  |           `-- rs0
  |               `-- log.1351p192
  |-- obiwan
  |   `-- 135
  |       `-- 1351p192
  |           `-- rs0
  |               |-- metacat-elg-1351p192.fits
  |               |-- sim_ids_added.fits
  |               `-- simcat-elg-1351p192.fits
  |-- metrics
  |   `-- 135
  |       `-- 1351p192
  |           `-- rs0
  |               |-- all-models-1351p192.fits
  |               `-- blobs-1351p192.fits.gz
  |-- tractor
  |   `-- 135
  |       `-- 1351p192
  |           `-- rs0
  |               |-- brick-1351p192.sha256sum
  |               `-- tractor-1351p192.fits
  `-- tractor-i
  |    `-- 135
  |        `-- 1351p192
  |            `-- rs0
  |                `-- tractor-1351p192.fits
  |-- coadd
  |   `-- 135
  |       `-- 1351p192
  |           `-- rs0
  |               |-- legacysurvey-1351p192-ccds.fits
  |               |-- legacysurvey-1351p192-chi2-g.fits.fz
  |               |-- legacysurvey-1351p192-chi2-r.fits.fz
  |               |-- legacysurvey-1351p192-chi2-z.fits.fz
  |               |-- legacysurvey-1351p192-image-g.fits.fz
  |               |-- legacysurvey-1351p192-image-r.fits.fz
  |               |-- legacysurvey-1351p192-image-z.fits.fz
  |               |-- legacysurvey-1351p192-image.jpg
  |               |-- legacysurvey-1351p192-invvar-g.fits.fz
  |               |-- legacysurvey-1351p192-invvar-r.fits.fz
  |               |-- legacysurvey-1351p192-invvar-z.fits.fz
  |               |-- legacysurvey-1351p192-model-g.fits.fz
  |               |-- legacysurvey-1351p192-model-r.fits.fz
  |               |-- legacysurvey-1351p192-model-z.fits.fz
  |               |-- legacysurvey-1351p192-model.jpg
  |               |-- legacysurvey-1351p192-resid.jpg
  |               |-- legacysurvey-1351p192-sims-g.fits.fz
  |               |-- legacysurvey-1351p192-sims-r.fits.fz
  |               |-- legacysurvey-1351p192-sims-z.fits.fz
  |               `-- legacysurvey-1351p192-simscoadd.jpg


The top level output directory includes the following files:

* The usual six directories: **tractor**, **tractor-i**, **coadd**, **metrics**, **checkpoint**, **logs**

* Monte Carlo simulation metadata directory: *obiwan*

Subdirectories follow the usual Data Relase format of **.../bri/brick/**, where **bri** is the first three letters of each brick.
The multiple iteractions per brick are identified by directories named **rs[0-9]+**, where the **rs** stands for the **Row** of the unique id-sorted table of randoms in the brick to **Start** from.
The **[0-9]+** is the index of that row, which would be 0, 500, 1000, etc. when 500 fake galaxies are added to the images in each iteration.

For example, the tractor catalogues containing the first 1500 fake sources in brick **1757p240** are in

- .../tractor/175/1757p240/**rs0**/
- .../tractor/175/1757p240/**rs500**/
- .../tractor/175/1757p240/**rs1000**/

However, there will be additional directories to **rs[0-9]+**, such as **skip_rs[0-9]+**, which contain the sources that were *skipped* because they were within 5 arcsec of another random.
There are actually four sets of directories like this:

- **rs[0-9]+**: initial set of iterations per brick
- **skip_rs[0-9]+**: skipped sources from the initial set of iterations
- **more_rs[0-9]+**: new set of iterations per brick for randoms that were added to the database after the run completed (e.g. more randoms were needed)
- **more_skip_rs[0-9]+**: skipped sources from the new set of iterations

Note that the coadd/ directory has a subset of the usual outputs because it is the directory with the largest file sizes.

The simulation metadata for each Monte Carlo simulation, stored in **.../obiwan/bri/brick/rs[0-9]+/**, consists of four files:

- **metacat-elg-brick.fits**: how obiwan was run (e.g. brickname)
- **simcat-elg-brick.fits**: truth table of the final properties of the sources added to the images (e.g. guassian noise and galactic extinction make the fluxes actually added different from those in the DB)
- **skippedids-elg-brick.fits**: ids of the randoms that were skipped because the within 5 arcsec of another random
- **sim_ids_added.fits**: unique ids of the randoms that overlap with at least one CCD so are actually added to the images

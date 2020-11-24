.. _user-running:

Running
=======

Executable
----------

As in **legacypipe**, :pyobiwan:`runbrick.py` is the (main) executable.
Type ``python obiwan/runbrick.py --help`` to print the command line arguments,
shown below. **legacypipe** arguments are listed first, then in a separate group are **Obiwan**-specific ones::

  runbrick.py started at 2020-11-24 04:15:03
  command-line args: ['obiwan/runbrick.py', '--help']
  usage: runbrick.py [-h] [-r RUN] [-f FORCE] [-F] [-s STAGE] [-n]
                     [-w WRITE_STAGE] [-v] [--checkpoint CHECKPOINT_FILENAME]
                     [--checkpoint-period CHECKPOINT_PERIOD] [-b BRICK]
                     [--radec RADEC RADEC] [--pixscale PIXSCALE] [-W WIDTH]
                     [-H HEIGHT] [--zoom ZOOM ZOOM ZOOM ZOOM] [-d OUTPUT_DIR]
                     [--release RELEASE] [--survey-dir SURVEY_DIR]
                     [--blob-mask-dir BLOB_MASK_DIR] [--cache-dir CACHE_DIR]
                     [--threads THREADS] [-p] [--plots2] [-P PICKLE_PAT]
                     [--plot-base PLOT_BASE] [--plot-number PLOT_NUMBER]
                     [--ceres] [--no-wise-ceres] [--nblobs NBLOBS] [--blob BLOB]
                     [--blobid BLOBID] [--blobxy BLOBXY BLOBXY]
                     [--blobradec BLOBRADEC BLOBRADEC]
                     [--max-blobsize MAX_BLOBSIZE] [--check-done] [--skip]
                     [--skip-coadd] [--skip-calibs] [--old-calibs-ok]
                     [--skip-metrics] [--nsigma NSIGMA]
                     [--saddle-fraction SADDLE_FRACTION]
                     [--saddle-min SADDLE_MIN] [--reoptimize] [--no-iterative]
                     [--no-wise] [--unwise-dir UNWISE_DIR]
                     [--unwise-tr-dir UNWISE_TR_DIR] [--galex]
                     [--galex-dir GALEX_DIR] [--early-coadds] [--blob-image]
                     [--no-lanczos] [--gpsf] [--no-hybrid-psf]
                     [--no-normalize-psf] [--apodize] [--coadd-bw]
                     [--bands BANDS] [--no-tycho] [--no-gaia]
                     [--no-large-galaxies] [--min-mjd MIN_MJD]
                     [--max-mjd MAX_MJD] [--no-splinesky] [--no-subsky]
                     [--no-unwise-coadds] [--no-outliers] [--cache-outliers]
                     [--bail-out] [--fit-on-coadds] [--no-ivar-reweighting]
                     [--no-galaxy-forcepsf] [--less-masking] [--ubercal-sky]
                     [--subsky-radii SUBSKY_RADII SUBSKY_RADII SUBSKY_RADII]
                     [--read-serial] [--log-fn LOG_FN] [--subset SUBSET]
                     [--ran-fn RAN_FN] [--fileid FILEID] [--rowstart ROWSTART]
                     [--nobj NOBJ] [--skipid SKIPID] [--col-radius COL_RADIUS]
                     [--sim-stamp {tractor,galsim}]
                     [--add-sim-noise {gaussian,poisson}] [--image-eq-model]
                     [--sim-blobs] [--seed SEED] [--ps PS] [--ps-t0 PS_T0]

  Main "Obiwan" script for the Legacy Survey (DECaLS, MzLS, Bok) data
  reductions.

  optional arguments:
    -h, --help            show this help message and exit
    -r RUN, --run RUN     Set the run type to execute
    -f FORCE, --force-stage FORCE
                          Force re-running the given stage(s) -- don't read from
                          pickle.
    -F, --force-all       Force all stages to run
    -s STAGE, --stage STAGE
                          Run up to the given stage(s)
    -n, --no-write
    -w WRITE_STAGE, --write-stage WRITE_STAGE
                          Write a pickle for a given stage: eg "tims",
                          "image_coadds", "srcs"
    -v, --verbose         Make more verbose
    --checkpoint CHECKPOINT_FILENAME
                          Write to checkpoint file?
    --checkpoint-period CHECKPOINT_PERIOD
                          Period for writing checkpoint files, in seconds;
                          default 600
    -b BRICK, --brick BRICK
                          Brick name to run; required unless --radec is given
    --radec RADEC RADEC   RA,Dec center for a custom location (not a brick)
    --pixscale PIXSCALE   Pixel scale of the output coadds (arcsec/pixel)
    -W WIDTH, --width WIDTH
                          Target image width, default 3600
    -H HEIGHT, --height HEIGHT
                          Target image height, default 3600
    --zoom ZOOM ZOOM ZOOM ZOOM
                          Set target image extent (default "0 3600 0 3600")
    -d OUTPUT_DIR, --outdir OUTPUT_DIR
                          Set output base directory, default "."
    --release RELEASE     Release code for output catalogs (default determined
                          by --run)
    --survey-dir SURVEY_DIR
                          Override the $LEGACY_SURVEY_DIR environment variable
    --blob-mask-dir BLOB_MASK_DIR
                          The base directory to search for blob masks during sky
                          model construction
    --cache-dir CACHE_DIR
                          Directory to search for cached files
    --threads THREADS     Run multi-threaded
    -p, --plots           Per-blob plots?
    --plots2              More plots?
    -P PICKLE_PAT, --pickle PICKLE_PAT
                          Pickle filename pattern, default
                          pickles/runbrick-%(brick)s-%%(stage)s.pickle
    --plot-base PLOT_BASE
                          Base filename for plots, default brick-BRICK
    --plot-number PLOT_NUMBER
                          Set PlotSequence starting number
    --ceres               Use Ceres Solver for all optimization?
    --no-wise-ceres       Do not use Ceres Solver for unWISE forced phot
    --nblobs NBLOBS       Debugging: only fit N blobs
    --blob BLOB           Debugging: start with blob #
    --blobid BLOBID       Debugging: process this list of (comma-separated) blob
                          ids.
    --blobxy BLOBXY BLOBXY
                          Debugging: run the single blob containing pixel <bx>
                          <by>; this option can be repeated to run multiple
                          blobs.
    --blobradec BLOBRADEC BLOBRADEC
                          Debugging: run the single blob containing RA,Dec <ra>
                          <dec>; this option can be repeated to run multiple
                          blobs.
    --max-blobsize MAX_BLOBSIZE
                          Skip blobs containing more than the given number of
                          pixels.
    --check-done          Just check for existence of output files for this
                          brick?
    --skip                Quit if the output catalog already exists.
    --skip-coadd          Quit if the output coadd jpeg already exists.
    --skip-calibs         Do not run the calibration steps
    --old-calibs-ok       Allow old calibration files (where the data validation
                          does not necessarily pass).
    --skip-metrics        Do not generate the metrics directory and files
    --nsigma NSIGMA       Set N sigma source detection thresh
    --saddle-fraction SADDLE_FRACTION
                          Fraction of the peak height for selecting new sources.
    --saddle-min SADDLE_MIN
                          Saddle-point depth from existing sources down to new
                          sources (sigma).
    --reoptimize          Do a second round of model fitting after all model
                          selections
    --no-iterative        Turn off iterative source detection?
    --no-wise             Skip unWISE forced photometry
    --unwise-dir UNWISE_DIR
                          Base directory for unWISE coadds; may be a colon-
                          separated list
    --unwise-tr-dir UNWISE_TR_DIR
                          Base directory for unWISE time-resolved coadds; may be
                          a colon-separated list
    --galex               Perform GALEX forced photometry
    --galex-dir GALEX_DIR
                          Base directory for GALEX coadds
    --early-coadds        Make early coadds?
    --blob-image          Create "imageblob" image?
    --no-lanczos          Do nearest-neighbour rather than Lanczos-3 coadds
    --gpsf                Use a fixed single-Gaussian PSF
    --no-hybrid-psf       Don't use a hybrid pixelized/Gaussian PSF model
    --no-normalize-psf    Do not normalize the PSF model to unix flux
    --apodize             Apodize image edges for prettier pictures?
    --coadd-bw            Create grayscale coadds if only one band is available?
    --bands BANDS         Set the list of bands (filters) that are included in
                          processing: comma-separated list, default "g,r,z"
    --no-tycho            Don't use Tycho-2 sources as fixed stars
    --no-gaia             Don't use Gaia sources as fixed stars
    --no-large-galaxies   Don't seed (or mask in and around) large galaxies.
    --min-mjd MIN_MJD     Only keep images taken after the given MJD
    --max-mjd MAX_MJD     Only keep images taken before the given MJD
    --no-splinesky        Use constant sky rather than spline.
    --no-subsky           Do not subtract the sky background.
    --no-unwise-coadds    Turn off writing FITS and JPEG unWISE coadds?
    --no-outliers         Do not compute or apply outlier masks
    --cache-outliers      Use outlier-mask file if it exists?
    --bail-out            Bail out of "fitblobs" processing, writing all blobs
                          from the checkpoint and skipping any remaining ones.
    --fit-on-coadds       Fit to coadds rather than individual CCDs (e.g., large
                          galaxies).
    --no-ivar-reweighting
                          Reweight the inverse variance when fitting on coadds.
    --no-galaxy-forcepsf  Do not force PSFs within galaxy mask.
    --less-masking        Turn off background fitting within MEDIUM mask.
    --ubercal-sky         Use the ubercal sky-subtraction (only used with --fit-
                          on-coadds and --no-subsky).
    --subsky-radii SUBSKY_RADII SUBSKY_RADII SUBSKY_RADII
                          Sky-subtraction radii: rmask, rin, rout [arcsec] (only
                          used with --fit-on-coadds and --no-subsky). Image
                          pixels r<rmask are fully masked and the pedestal sky
                          background is estimated from an annulus rin<r<rout on
                          each CCD centered on the targetwcs.crval coordinates.
    --read-serial         Read images in series, not in parallel?
    --log-fn LOG_FN       Log to given filename instead of stdout
    --ps PS               Run "ps" and write results to given filename?
    --ps-t0 PS_T0         Unix-time start for "--ps"

  Obiwan:
    Obiwan-specific arguments

    --subset SUBSET       COSMOS subset number [0 to 4, 10 to 12], only used if
                          --run cosmos
    --ran-fn RAN_FN       Randoms filename; if not provided, run equivalent to
                          legacypipe.runbrick
    --fileid FILEID       Index of ran-fn
    --rowstart ROWSTART   Zero indexed, row of ran-fn, after it is cut to brick,
                          to start on
    --nobj NOBJ           Number of objects to inject in the given brick; if -1,
                          all objects in ran-fn are added
    --skipid SKIPID       Inject collided objects from ran-fn of previous
                          skipid-1 run. In this case, no cut based on --nobj and
                          --rowstart is applied.
    --col-radius COL_RADIUS
                          Collision radius in arcseconds, used to define
                          collided simulated objects. Ignore if negative
    --sim-stamp {tractor,galsim}
                          Method to simulate objects
    --add-sim-noise {gaussian,poisson}
                          Add noise from the simulated source to the image.
    --image-eq-model      Set image ivar by model only (ignore real image ivar)?
    --sim-blobs           Process only the blobs that contain simulated sources
    --seed SEED           Random seed to add noise to injected sources of ran-
                          fn.

  e.g., to run a small field containing a cluster: python -u obiwan/runbrick.py
  --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-
  cluster-%%s.pickle

.. note::

  :pyobiwan:`runbrick.py` can be run from the command line or from a python script

    .. code-block:: python

      from obiwan import runbrick
      runbrick.main(args)

  with arguments ``args``, as examplified in :root:`bin/mpi_main_runbricks.py`.

Scripts
-------

Some scripts are available in the :root:`bin` directory:

* :root:`bin/runbrick.sh` to run a single brick, which can be easily modified to launch on a batch system.

* :root:`bin/mpi_runbricks.sh` to run bricks on several MPI ranks (can also be used without MPI).

.. note::

  The **legacypipe** environment variables are defined in :root:`bin/legacypipe-env.sh`.
  and **Obiwan** settings (e.g. bricks to run) in :root:`bin/settings.py`.

.. note::

  The ``SURVEY_DIR`` directory should contain the directory ``images``, ``calib`` (if you not wish to rerun them),
  ``ccds-annotated-*`` and ``survey-*`` files.

On your laptop
^^^^^^^^^^^^^^

``runbrick.sh`` can be run within **Docker** through (``chmod u+x mpi_runbricks.sh`` if necessary)::

  docker run --volume $HOME:/homedir/ --image={dockerimage} ./mpi_runbricks.sh

``mpi_runbricks.sh`` can be run similarly; just add ``mpiexec`` or ``mpirun`` in front.

On NERSC
^^^^^^^^

:root:`bin/runbrick.sh`::

  shifter --volume $HOME:/homedir/ --image={dockerimage} ./mpi_runbricks.sh

:root:`bin/mpi_runbricks.sh`, without MPI::

  shifter --volume $HOME:/homedir/ --image={dockerimage} ./mpi_runbricks.sh

or with 2 MPI tasks::

  srun -n 2 shifter --module=mpich-cle6 --volume $HOME:/homedir/ --image={dockerimage} ./mpi_runbricks.sh

.. note::

  By default, :root:`bin/mpi_runbricks.sh` uses your current **Obiwan** directory. To rather use the official release in the **Docker** image (``/src/obiwan``),
  uncomment ``export PYTHONPATH=...`` in :root:`bin/mpi_runbricks.sh`.

.. note::

  By default, :root:`bin/mpi_runbricks.sh` launches :root:`bin/mpi_main_runbricks.py` (which directly runs :pyobiwan:`runbrick.py`).
  To rather use :root:`bin/mpi_script_runbricks.sh` (which calls :pyobiwan:`bin/runbrick.sh`), pass the option ``-s``.

.. note::

  By default, :root:`bin/mpi_runbricks.sh` runs 8 threads OpenMP threads. You can change that using the ``OMP_NUM_THREADS`` environment variable.

Running
#######

As in **legacypipe**, ``py\obiwan\runbrick.py`` is the executable.
Type ``python py\obiwan\runbrick.py --help`` to print the command line arguments,
shown below. **legacypipe** arguments are listed first, then in a separate group are those
of **Obiwan**::

  runbrick.py starting at 2020-11-10T11:59:22.296158
  legacypipe git version: DR9.6.5-4-gbb698724
  Command-line args: ['py/obiwan/runbrick.py', '--help']
  python py/obiwan/runbrick.py --help

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
                     [--no-unwise-coadds] [--no-outliers] [--bail-out]
                     [--fit-on-coadds] [--no-ivar-reweighting]
                     [--no-galaxy-forcepsf] [--less-masking] [--ubercal-sky]
                     [--subsky-radii SUBSKY_RADII SUBSKY_RADII SUBSKY_RADII]
                     [--read-serial] [--subset SUBSET] [--ran-fn RAN_FN]
                     [--fileid FILEID] [--rowstart ROWSTART] [--nobj NOBJ]
                     [--skipid SKIPID] [--col-radius COL_RADIUS]
                     [--sim-stamp {tractor,galsim}] [--add-sim-noise]
                     [--image-eq-model] [--sim-blobs] [--seed SEED] [--ps PS]
                     [--ps-t0 PS_T0]

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
    --add-sim-noise       Add noise to simulated sources?
    --image-eq-model      Set image ivar by model only (ignore real image ivar)?
    --sim-blobs           Process only the blobs that contain simulated sources
    --seed SEED           Random seed to add noise to injected sources of ran-
                          fn.

  e.g., to run a small field containing a cluster: python -u obiwan/runbrick.py
  --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-
  cluster-%%s.pickle

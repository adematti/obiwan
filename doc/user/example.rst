.. _user-example:

Example on NERSC
================

First, create your **Obiwan** directory and copy/link the **legacy survey** data::

  mkdir -p ${CSCRATCH}/Obiwan/dr9/data/
  cp {legacysurveyroot}ccds-annotated-* ${CSCRATCH}/Obiwan/dr9/data/
  cp {legacysurveyroot}survey-* ${CSCRATCH}/Obiwan/dr9/data/
  ln -s {legacysurveyroot}calib/ ${CSCRATCH}/Obiwan/dr9/data/
  ln -s {legacysurveyroot}images/ ${CSCRATCH}/Obiwan/dr9/data/

Next, clone the :root:`Obiwan repo` and pull the docker image (see :ref:`user-building`)::

  cd
  git clone {gitrepo}
  shifterimg -v pull {dockerimage}

.. note::

  **Obiwan** executable is in the Docker image. In the following we just use :root:`bin`.

Generate randoms file (see :ref:`user-pre-processing`)::

  cd obiwan/bin
  shifter --volume ${HOME}:/homedir/ --image={dockerimage} /bin/bash
  source legacypipe-env.sh
  python preprocess.py --do randoms

Create run list, taking into account **legacypipe** versions used for bricks in ``$LEGACYPIPE_SURVEY_DIR/north/`` (see :ref:`user-running`)::

  python /src/obiwan/py/obiwan/scripts/runlist.py --outdir $LEGACYPIPE_SURVEY_DIR/north/ --brick bricklist.txt --write-list runlist.txt --modules legacypipe

Then run **Obiwan** (see :ref:`user-running`)::

  srun -n 2 shifter --module=mpich-cle6 --volume ${HOME}:/homedir/ --image={dockerimage} ./mpi_runbricks.sh

Enter your shifter image, check everything ran, match and plot the comparison (see :ref:`user-post-processing`)::

  shifter --volume ${HOME}:/homedir/ --image={dockerimage} /bin/bash
  cd ${HOME}/obiwan/bin
  python /src/obiwan/py/obiwan/scripts/check.py --outdir $CSCRATCH/Obiwan/dr9/test --brick bricklist_400N-EBV.txt
  python /src/obiwan/py/obiwan/scripts/match.py --cat-dir $CSCRATCH/Obiwan/dr9/test/merged --outdir $CSCRATCH/Obiwan/dr9/test --plot-hist plots/hist.png

You can also merge catalogs, plot cpu and memory usage, image cutouts::

  python /src/obiwan/py/obiwan/scripts/merge.py --filetype randoms tractor --cat-dir $CSCRATCH/Obiwan/dr9/test/merged --outdir $CSCRATCH/Obiwan/dr9/test
  python /src/obiwan/py/obiwan/scripts/resources.py --outdir $CSCRATCH/Obiwan/dr9/test --plot-fn plots/resources-summary.png
  python /src/obiwan/py/obiwan/scripts/cutout.py --outdir $CSCRATCH/Obiwan/dr9/test --plot-fn "plots/cutout_%(brickname)s-%(icut)d.png" --ncuts 2

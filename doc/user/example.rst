.. _example:

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

Generate randoms file (see :ref:`user-pre-processing`)::

  cd obiwan/bin
  shifter --volume ${HOME}:/homedir/ --image={dockerimage} python preprocess.py --do randoms

Then run **Obiwan** (see :ref:`user-running`)::

  srun -n 2 shifter --module=mpich-cle6 --volume ${HOME}:/homedir/ --image={dockerimage} ./mpi_runbricks.sh

Enter your **shifter** image, check everything ran, match and plot the comparison (see :ref:`user-post-processing`)::

  shifter --volume ${HOME}:/homedir/ --image={dockerimage} /bin/bash
  cd ${HOME}/obiwan/bin
  python ../py/obiwan/scripts/check.py --outdir $CSCRATCH/Obiwan/dr9/test --brick bricklist_400N-EBV.txt
  python ../py/obiwan/scripts/match.py --cat-dir $CSCRATCH/Obiwan/dr9/test/merged --outdir $CSCRATCH/Obiwan/dr9/test --plot-hist plots/hist.png

You can also merge catalogs, plot cpu and memory usage, image cutouts::

  python ../py/obiwan/scripts/merge.py --filetype randoms tractor --cat-dir $CSCRATCH/Obiwan/dr9/test/merged --outdir $CSCRATCH/Obiwan/dr9/test
  python ../py/obiwan/scripts/resources.py --outdir $CSCRATCH/Obiwan/dr9/test --plot-fn plots/resources-summary.png
  python ../py/obiwan/scripts/cutout.py --outdir $CSCRATCH/Obiwan/dr9/test --plot-fn "plots/cutout_%(brickname)s-%(icut)d.png" --ncuts 2

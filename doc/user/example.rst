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

Enter your **shifter** image, match input randoms file to **Tractor** output and plot the comparison (see :ref:`user-post-processing`)::

  shifter --volume ${HOME}:/homedir/ --image={dockerimage} /bin/bash
  cd ${HOME}/obiwan/bin
  python postprocess.py --do match
  python postprocess.py --do plot

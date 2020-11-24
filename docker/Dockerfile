FROM legacysurvey/legacypipe:DR9.6.7

WORKDIR /src/

RUN apt -y update && apt install -y apt-utils && echo yes

RUN DEBIAN_FRONTEND=noninteractive \
  apt install -y --no-install-recommends \
  openmpi-bin \
  mpich \
  gettext \
  texinfo \
  gawk \
  libeigen3-dev \
  cmake \
  # # Remove APT files
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget -nv http://www.fftw.org/fftw-3.3.8.tar.gz \
  && tar xvzf fftw-3.3.8.tar.gz \
  && cd fftw-3.3.8 \
  && ./configure --enable-shared \
  && make -j4 \
  && make install \
  && make clean \
  && cd .. && rm -Rf fftw-3.3.8 fftw-3.3.8.tar.gz

RUN for x in \
  mpi4py \
  pandas \
  pytest \
  coverage \
  sphinx \
  sphinx-rtd-theme \
  galsim \
  ; do pip install $x; done \
  && rm -Rf /root/.cache/pip

# desiutil
RUN git clone https://github.com/desihub/desiutil.git desiutil \
  && cd desiutil \
  && python3 setup.py install

RUN rm -rf /src/legacypipe/py/obiwan

# DUST maps
RUN mkdir -p dust/maps \
  && cd dust/maps \
  && wget -c https://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_ngp.fits \
  && wget -c https://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_sgp.fits
ENV DUST_DIR=/src/dust

COPY . /src/obiwan
ENV PYTHONPATH /src/obiwan/py:${PYTHONPATH}

#RUN cd /src/obiwan/ \
#  && python3 setup.py install --home=/usr/local

WORKDIR /homedir/

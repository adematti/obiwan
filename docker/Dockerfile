FROM legacysurvey/legacypipe:DR9.6.5b

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
    h5py \
    pandas \
    pytest \
    sphinx \
    sphinx-rtd-theme \
    galsim \
    ; do pip install $x; done \
    && rm -Rf /root/.cache/pip

# desiutil
RUN cd /src/ \
    && git clone https://github.com/desihub/desiutil.git desiutil \
    && cd desiutil \
    && python3 setup.py install

RUN cd /src/ \
    && rm -rf legacypipe/py/obiwan

ADD . /src/obiwan

RUN cd /src/obiwan/ \
    && python3 setup.py install --home=/usr/local

FROM continuumio/miniconda3:24.11.1-0

WORKDIR /
RUN git clone https://github.com/scipy/scipy.git
WORKDIR /scipy
RUN git checkout v1.15.1
RUN git submodule update --init
RUN conda env create -f environment.yml
COPY custom_scipy/_ppoly_v1.15.1.pyx /scipy/scipy/interpolate/_ppoly.pyx
RUN echo 'conda activate scipy-dev' >> ~/.bashrc
RUN echo '. /opt/conda/etc/profile.d/conda.sh && conda activate scipy-dev && pip install -e . --no-build-isolation' | bash

COPY *.py /bottlemod/
COPY createfigures.sh /bottlemod/
WORKDIR /bottlemod
CMD ./createfigures.sh

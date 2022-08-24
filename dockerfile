FROM continuumio/miniconda3

RUN apt update
RUN apt install -y git

WORKDIR /
RUN git clone https://github.com/scipy/scipy.git
WORKDIR /scipy
RUN git checkout 2c72ae27e59e64f47d64bf65800a291ff972df86
RUN conda env create -f environment.yml
RUN echo 'conda activate scipy-dev' >> ~/.bashrc
RUN git submodule update --init
COPY custom_scipy/_ppoly.pyx /scipy/scipy/interpolate/
RUN echo '. /opt/conda/etc/profile.d/conda.sh && conda activate scipy-dev && pip install .' | bash
RUN echo 'pip install matplotlib' | bash
WORKDIR /
RUN rm -rf scipy

COPY *.py /bottlemod/
COPY createfigures.sh /bottlemod/
WORKDIR /bottlemod
CMD ./createfigures.sh

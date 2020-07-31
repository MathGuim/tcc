FROM continuumio/miniconda3:4.6.14

ENV PROJECT_DIR /usr/local

WORKDIR ${PROJECT_DIR}

COPY environment.yml ${PROJECT_DIR}/

RUN conda env create -f environment.yml
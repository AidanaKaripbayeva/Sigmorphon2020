FROM pytorch/pytorch:latest
MAINTAINER bjlunt2@illinois.edu

USER root

RUN conda install -y jupyterlab matplotlib
RUN conda install -y nltk keras tensorflow

WORKDIR /tmp
RUN git clone https://github.com/IBM/pytorch-seq2seq.git && \
	cd pytorch-seq2seq && pip install . && \
	cd /tmp && rm -rf pytorch-seq2seq

WORKDIR /workspace

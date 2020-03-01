FROM pytorch/pytorch:latest
MAINTAINER bjlunt2@illinois.edu

USER root

RUN conda config --add channels conda-forge && \
	conda config --set channel_priority strict
RUN conda install -y jupyterlab matplotlib pandas
RUN conda install -y nltk keras tensorflow

RUN mkdir /sigmorphon2020 && \
		cd /sigmorphon2020 && \
		git clone https://github.com/sigmorphon2020/task0-data.git && \
		git clone https://github.com/sigmorphon2020/task0-baselines.git && \
		chmod -R 755 /sigmorphon2020

RUN mkdir /software
WORKDIR /software
RUN git clone https://github.com/bharathgs/Awesome-pytorch-list.git
RUN	git clone https://github.com/pytorch/fairseq.git
RUN	( git clone https://github.com/pytorch/text.git && cd text && pip install .)
RUN ( git clone https://github.com/IBM/pytorch-seq2seq.git && \
	cd pytorch-seq2seq && pip install . )


RUN useradd -ms /bin/bash devuser

USER devuser
WORKDIR /home/devuser

EXPOSE 8888
CMD ["jupyter","lab","--no-browser", "--ip=0.0.0.0","--port=8888"]

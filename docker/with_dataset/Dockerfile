FROM cs546_turkic_base
MAINTAINER bjlunt2@illinois.edu

USER root

RUN mkdir /sigmorphon2020 && \
		cd /sigmorphon2020 && \
		git clone https://github.com/sigmorphon2020/task0-data.git && \
		git clone https://github.com/sigmorphon2020/task0-baselines.git && \
		git clone https://github.com/unimorph/um-canonicalize.git && \
		chmod -R 755 /sigmorphon2020

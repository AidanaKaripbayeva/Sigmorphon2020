FROM cs546_turkic_dataset
MAINTAINER bjlunt2@illinois.edu

USER root
RUN useradd -ms /bin/bash devuser

USER devuser
WORKDIR /home/devuser
RUN ln -s /mnt/shared shared

EXPOSE 8888
CMD ["jupyter","lab","--no-browser", "--ip=0.0.0.0","--port=8888"]

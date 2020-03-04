#!/bin/bash

###Snippet from http://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
###end snippet

pushd ${SCRIPT_DIR}

docker build -t cs546_turkic_base ./base_software
docker build --no-cache -t cs546_turkic_dataset ./with_dataset
docker build -t luntlab/cs546_turkic_user:latest -t cs546_turkic_user ./user_image

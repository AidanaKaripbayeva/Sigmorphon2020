#!/bin/bash

###Snippet from http://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
###end snippet


CONTAINER_ID=$( docker run -d -P cs546_turkic_base )
CONTAINER_NAME=$( docker inspect --format '{{ .Name }}' ${CONTAINER_ID} )
HOST_PORT=$( docker inspect --format '{{ (index (index .NetworkSettings.Ports "8888/tcp") 0).HostPort }}' ${CONTAINER_ID})
#CONTAINER_IP=$( docker inspect --format '{{ .NetworkSettings.IPAddress }}' ${CONTAINER_ID} )

#get token
sleep 2 #this still introduces a race condition. Should use Expect

#no race condition. Can't get it to work right.
#docker logs ${CONTAINER_ID} 2>&1 | expect << EOF
#expect token
#exit
#EOF

#could have probably picked up the token in the last command. Meh.
CONNECT_TOKEN=$( docker logs ${CONTAINER_ID} 2>&1 | grep token | tail -n 1 | sed 's/.*token=//' )

echo "Connecting to ${CONTAINER_NAME} "

open "http://localhost:${HOST_PORT}/?token=${CONNECT_TOKEN}"
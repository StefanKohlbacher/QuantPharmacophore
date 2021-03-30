#!/bin/bash

IMAGE_NAME="qphar:latest"
CONTAINER_NAME="qphar-container"


# build docker container
docker build -t "$IMAGE_NAME" .
docker rm "$CONTAINER_NAME"
docker run -dit -P --name "$CONTAINER_NAME" -v ~/container_data:/qphar/data "$IMAGE_NAME"
docker exec "$CONTAINER_NAME" sh /qphar/prepare_docker_environment.sh
docker attach "$CONTAINER_NAME"
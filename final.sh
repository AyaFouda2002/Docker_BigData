#!/bin/bash

# The container ID or name
CONTAINER_ID="bd-a1"
CONTAINER_DIR="/home/doc-bd-a1/"

# The directory where to copy the files on the host
HOST_DIR="./service-result/"

# Create output folder
mkdir -p "$HOST_DIR"

# Copy .txt and .png files from the container to the host.
# This command assumes that the files are located at the root of the container's filesystem.
docker cp "$CONTAINER_ID":"$CONTAINER_DIR"k.txt "$HOST_DIR"
docker cp "$CONTAINER_ID":"$CONTAINER_DIR"eda-in-1.txt "$HOST_DIR"
docker cp "$CONTAINER_ID":"$CONTAINER_DIR"eda-in-2.txt "$HOST_DIR"
docker cp "$CONTAINER_ID":"$CONTAINER_DIR"eda-in-3.txt "$HOST_DIR"
docker cp "$CONTAINER_ID":"$CONTAINER_DIR"vis.png "$HOST_DIR"

# Stop the container
docker stop "$CONTAINER_ID"
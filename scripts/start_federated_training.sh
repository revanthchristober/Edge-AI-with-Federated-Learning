#!/bin/bash

# Script to start federated training using Syft TensorFlow

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Default number of client nodes
NUM_CLIENTS=3

# Parse optional argument for number of clients
while getopts n: flag
do
    case "${flag}" in
        n) NUM_CLIENTS=${OPTARG};;
    esac
done

# Build Docker containers for server and clients
echo "Building Docker containers for federated learning..."
docker-compose build

# Start server container
echo "Starting server for federated learning..."
docker-compose up -d server

# Start client containers
echo "Starting $NUM_CLIENTS clients for federated learning..."
for i in $(seq 1 $NUM_CLIENTS)
do
    CLIENT_NAME="client_${i}"
    echo "Starting Client $CLIENT_NAME..."
    docker-compose up -d $CLIENT_NAME
done

# Wait for clients to start
sleep 10

# Monitor logs for server and clients
echo "Monitoring logs for server and clients..."
docker-compose logs -f server client_1 client_2 client_3

# Stop all containers after training is complete
echo "Federated training complete. Stopping all containers..."
docker-compose down

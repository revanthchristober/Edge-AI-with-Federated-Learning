#!/bin/bash

# Script for deploying the federated learning system locally or on cloud

# Check for Docker and Docker Compose installation
if ! command -v docker &> /dev/null
then
    echo "Docker not found! Please install Docker first."
    exit
fi

if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose not found! Please install Docker Compose first."
    exit
fi

# Set up Docker environment variables
export DOCKER_BUILDKIT=1

# Pull latest base images
docker pull tensorflow/tensorflow:2.10.0-gpu

# Build Docker containers for server and clients
echo "Building Docker images for federated learning..."
docker-compose build

# Deploy the system
echo "Starting federated learning system..."
docker-compose up -d

# Check if the deployment was successful
if [ $? -eq 0 ]; then
    echo "Federated learning system deployed successfully."
    echo "Server running at http://localhost:5000"
else
    echo "Error deploying federated learning system."
    exit 1
fi

# Function to clean up the Docker environment
cleanup() {
    echo "Stopping and removing all containers..."
    docker-compose down
    echo "Federated learning system stopped."
}

# Check if user wants to stop the deployment
while true; do
    read -p "Do you want to stop the federated learning system? (yes/no): " yn
    case $yn in
        [Yy]* ) cleanup; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

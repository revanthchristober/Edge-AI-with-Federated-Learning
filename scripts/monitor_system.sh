#!/bin/bash

# Script to monitor system performance and resource usage during federated learning

# Function to check if a command exists
function check_command() {
    if ! command -v $1 &> /dev/null
    then
        echo "Error: $1 is not installed. Please install $1 and try again."
        exit 1
    fi
}

# Check for necessary system monitoring tools
check_command nvidia-smi
check_command htop
check_command free

# Set default refresh interval in seconds
INTERVAL=5

# Parse optional argument for interval
while getopts i: flag
do
    case "${flag}" in
        i) INTERVAL=${OPTARG};;
    esac
done

echo "Monitoring system performance... (Press Ctrl+C to stop)"
echo "Refresh interval: $INTERVAL seconds"

# Start monitoring system performance
while true
do
    echo "-------------------------------------------"
    echo "Timestamp: $(date)"
    echo "-------------------------------------------"

    echo "GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
    echo

    echo "CPU and Memory Usage:"
    htop -d $INTERVAL
    echo

    echo "Free Memory:"
    free -h
    echo

    sleep $INTERVAL
done

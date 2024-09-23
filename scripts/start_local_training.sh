#!/bin/bash

# Script to start local training on client nodes

# Check if Syft and TensorFlow are installed
if ! command -v syft &> /dev/null || ! python3 -c "import tensorflow" &> /dev/null
then
    echo "Error: Syft and TensorFlow are not installed. Exiting..."
    exit 1
fi

# Set the default values for training parameters
BATCH_SIZE=64
EPOCHS=10
LEARNING_RATE=0.001
CLIENT_ID="client_1"
DATA_PATH="data/local/client_${CLIENT_ID}/"
MODEL_PATH="models/client_${CLIENT_ID}/"
LOG_PATH="logs/client_${CLIENT_ID}/"

# Create necessary directories if not present
mkdir -p $DATA_PATH $MODEL_PATH $LOG_PATH

# Parse optional arguments for batch size, epochs, and learning rate
while getopts b:e:l:c: flag
do
    case "${flag}" in
        b) BATCH_SIZE=${OPTARG};;
        e) EPOCHS=${OPTARG};;
        l) LEARNING_RATE=${OPTARG};;
        c) CLIENT_ID=${OPTARG};;
    esac
done

# Update paths based on client ID
DATA_PATH="data/local/client_${CLIENT_ID}/"
MODEL_PATH="models/client_${CLIENT_ID}/"
LOG_PATH="logs/client_${CLIENT_ID}/"

# Run local training script
echo "Starting local training for Client ${CLIENT_ID} with Batch Size: ${BATCH_SIZE}, Epochs: ${EPOCHS}, Learning Rate: ${LEARNING_RATE}"
python3 training/local_training.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --log_path $LOG_PATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE

# Check if the training succeeded
if [ $? -eq 0 ]; then
    echo "Local training for Client ${CLIENT_ID} completed successfully."
else
    echo "Error occurred during local training for Client ${CLIENT_ID}. Check the logs for details."
    exit 1
fi

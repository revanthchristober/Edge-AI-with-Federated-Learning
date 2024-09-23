# Base Image: Use TensorFlow image with Python 3.8
FROM tensorflow/tensorflow:2.10.0-gpu

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Install Miniconda for managing environments
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create a conda environment and install Python libraries
RUN conda create -n federated_learning python=3.8
RUN echo "source activate federated_learning" > ~/.bashrc

# Install necessary Python packages in the conda environment
RUN conda install -n federated_learning -c conda-forge tensorflow \
    syft \
    numpy \
    pandas \
    h5py \
    flask \
    requests \
    pillow \
    pip && \
    conda clean -a

# Copy project files
WORKDIR /app
COPY . /app

# Expose default Flask port and run app
EXPOSE 5000

# Run server or client node based on the entrypoint
CMD ["bash"]

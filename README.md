# **EdgeAI-FederatedLearning with Syft and TensorFlow**

This project implements a **federated learning system** for **Edge AI**, using **PySyft** and **TensorFlow** to train models on decentralized devices while preserving data privacy. Each device (client) trains the model locally on its data, and only the model updates (parameters) are shared with a central server for aggregation.

---

## **Project Structure**

```bash
EdgeAI-FederatedLearning/
├── data/                                 # Data-related assets and preprocessing
│   ├── raw/                              # Raw, unprocessed data
│   ├── processed/                        # Preprocessed data for training
│   ├── download_data.py                  # Script for downloading and setting up the dataset
│   └── preprocess_data.py                # Script for data preprocessing
├── models/                               # Model architectures
│   ├── cnn_model.py                      # Convolutional neural network model architecture
│   ├── lstm_model.py                     # LSTM model architecture for time-series or NLP data
│   └── model_utils.py                    # Utility functions for model management
├── training/                             # Training scripts
│   ├── local_training.py                 # Script for local training on client nodes
│   ├── federated_training.py             # Script for federated training setup
│   ├── aggregation.py                    # Federated model aggregation logic (e.g., FedAvg)
│   └── evaluate.py                       # Script for evaluating the model
├── federated_system/                     # Federated learning infrastructure and Syft integration
│   ├── client_node.py                    # Client node training script
│   ├── server_node.py                    # Central server for parameter aggregation
│   └── federated_manager.py              # Manager for communication and orchestration between nodes
├── config/                               # Configuration files
│   ├── hyperparameters.yaml              # Hyperparameters for model training
│   ├── federated_config.yaml             # Configuration for federated learning
│   └── environment.yaml                  # Environment setup file
├── evaluation/                           # Evaluation scripts and metrics
│   ├── evaluate_model.py                 # Script to evaluate model performance
│   ├── metrics.py                        # Custom metrics for evaluation (precision, recall, etc.)
│   └── visualization.py                  # Visualization tools for model performance
├── utils/                                # Utility functions
│   ├── data_utils.py                     # Helper functions for data management
│   ├── logging_utils.py                  # Functions to handle structured logging
│   └── federated_utils.py                # Utility functions for Syft setup and communication
├── deployment/                           # Deployment-related files
│   ├── Dockerfile                        # Dockerfile for containerizing the system
│   ├── docker-compose.yml                # Multi-container setup for federated learning
│   └── federated_deployment.sh           # Script for deployment on cloud or local machines
├── tests/                                # Unit and integration tests
│   ├── unit_tests/
│   │   ├── test_model.py                 # Unit tests for model architecture
│   │   ├── test_training.py              # Unit tests for training
│   │   └── test_utils.py                 # Unit tests for utility functions
│   └── integration_tests/
│       ├── test_federated_training.py    # Integration tests for the federated training pipeline
│       ├── test_local_training.py        # Integration tests for local training
│       └── test_system_performance.py    # End-to-end performance tests
├── scripts/                              # Bash scripts for managing experiments
│   ├── start_local_training.sh           # Script to run local training
│   ├── start_federated_training.sh       # Script to start federated training
│   └── monitor_system.sh                 # Script to monitor system performance
├── docs/                                 # Documentation
│   ├── project_overview.md               # Project overview, objectives, and scope
│   ├── federated_architecture.md         # Detailed documentation on federated learning architecture
│   ├── model_architecture.md             # Documentation for model architectures used
│   ├── evaluation_strategy.md            # Explanation of evaluation metrics and tracking
│   └── deployment_guide.md               # Guide for deploying the federated learning system
└── README.md                             # Setup instructions, usage, and deployment guide
```

---

## **1. Setup Instructions**

### **1.1 Clone the Repository**

```bash
git clone https://github.com/your-username/EdgeAI-FederatedLearning.git
cd EdgeAI-FederatedLearning
```

### **1.2 Install Dependencies**

Make sure to install the required libraries for both **TensorFlow** and **Syft**.

```bash
pip install -r requirements.txt
```

Content of `requirements.txt`:

```
tensorflow==2.6.0
syft==0.6.0
numpy
pandas
matplotlib
scikit-learn
```

### **1.3 Download the Dataset**

Run the following script to download and preprocess the **FEMNIST** dataset for federated learning:

```bash
python data/download_data.py --iid --sf 1.0 -k 0 -t sample --tf 0.9
```

### **1.4 Preprocessing the Data**

After downloading, you can preprocess the data using the following script:

```bash
python data/preprocess_data.py
```

This will prepare the dataset for local and federated learning.

---

## **2. Local Training**

You can perform local training on a single client node using the following script.

### **Run Local Training**

```bash
python training/local_training.py
```

### **Hyperparameters**

The **hyperparameters** can be configured in the **config/hyperparameters.yaml** file.

Example **hyperparameters.yaml**:

```yaml
learning_rate: 0.001
epochs: 10
batch_size: 64
model: "cnn"   # Options: cnn, lstm
```

---

## **3. Federated Learning Setup**

### **3.1 Running the Federated Training System**

In a federated learning scenario, you will need to start the **server node** and multiple **client nodes**.

1. **Server Node**:

```bash
python federated_system/server_node.py
```

2. **Client Nodes** (Run this on multiple machines or containers):

```bash
python federated_system/client_node.py
```

The **federated_manager.py** script manages the orchestration of client and server nodes and handles the communication between them.

### **3.2 Federated Configuration**

Edit the **config/federated_config.yaml** to set up the number of clients, aggregation strategy, and communication parameters.

Example **federated_config.yaml**:

```yaml
clients:
  total_clients: 10
  clients_per_round: 5
aggregation:
  strategy: "fedavg"
communication:
  rounds: 20
  server_address: "localhost:8000"
```

### **3.3 Model Aggregation**

The **aggregation.py** script contains the logic for aggregating model updates from clients (e.g., **Federated Averaging - FedAvg**).

---

## **4. Evaluation**

### **4.1 Evaluating the Model**

Once training is completed, you can evaluate the global model by running:

```bash
python training/evaluate.py
```

This script evaluates the model on the test dataset and provides metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

### **4.2 Visualization**

The **evaluation/visualization.py** script can be used to plot accuracy, loss curves, and other metrics over time.

```bash
python evaluation/visualization.py
```

---

## **5. Deployment Guide**

The project is containerized using **Docker** and can be deployed across multiple machines or cloud instances. Refer to **deployment/deployment_guide.md** for the complete deployment instructions.

### **5.1 Docker Setup**

Build the Docker image:

```bash
docker build -t federated-learning .
```

### **5.2 Docker Compose for Multi-Node Deployment**

Use **Docker Compose** to launch the server and client nodes.

```bash
docker-compose up --build
```

This will launch both the **server** and multiple **client nodes** defined in the `docker-compose.yml`.

### **5.3 Deploying on Cloud (AWS EC2)**

Refer to **deployment_guide.md** for instructions on deploying the system on cloud environments like **AWS** or **GCP**.

---

## **6. Monitoring and Logs**

You can monitor system performance using Docker or custom logging. The **scripts/monitor_system.sh** script provides real-time system resource usage.

```bash
bash scripts/monitor_system.sh
```

### **Structured Logs**

All logs are structured and saved using the **utils/logging_utils.py** file. Logs include training progress, client communication, and model performance.

---

## **7. Tests**

### **7.1 Running Unit Tests**

You can run unit tests to verify the model architecture, training scripts, and utilities.

```bash
pytest tests/unit_tests/
```

### **7.2 Running Integration Tests**

For integration tests, such as federated training across nodes:



```bash
pytest tests/integration_tests/
```

---

## **Contributing**

Feel free to submit **pull requests** and raise **issues** to improve this project. Contributions are highly appreciated.

---

## **License**

This project is licensed under The GNU General Public License v3.0 License. See the [LICENSE](./LICENSE) file for more details.

---
# **Deployment Guide for Federated Learning System**

## **Introduction**

This guide provides a step-by-step process to deploy the **federated learning system** in a distributed environment using **PySyft** and **TensorFlow**. The deployment setup includes multiple **client nodes** that perform local training and a **central server** that aggregates the parameters from the client nodes.

The deployment strategy leverages **Docker** and **Ray.io** to scale federated learning across multiple nodes either in a **cloud environment** (e.g., AWS, Azure, GCP) or on **local machines**.

### **Pre-requisites**:

1. **Docker** and **Docker Compose** installed on all participating machines.
2. **Syft** and **TensorFlow** installed on both client and server nodes.
3. **Ray** for distributed communication and orchestration.
4. **Python 3.8+**.
5. **A cloud provider account** (if deploying on AWS, GCP, etc.).

---

## **1. Setting Up Docker Environment**

Docker is used to containerize the federated system. Each **client node** and the **central server** will run inside a Docker container.

### **1.1 Dockerfile for Client and Server Nodes**

Here is the **Dockerfile** for both client and server nodes:

```dockerfile
# Dockerfile for Federated Learning Node (Client/Server)

FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow
RUN pip install tensorflow

# Install PySyft
RUN pip install syft

# Copy the project files
COPY . .

# Expose ports
EXPOSE 8000

# Command to run the node (server or client)
CMD ["python", "start_node.py"]
```

### **1.2 `requirements.txt` File**

The **requirements.txt** file contains the dependencies for Syft, TensorFlow, and other necessary libraries:

```
tensorflow==2.6.0
syft==0.6.0
numpy
pandas
matplotlib
ray
```

### **1.3 Building Docker Images**

Once you have the **Dockerfile** and **requirements.txt** ready, build the image:

```bash
docker build -t federated-node .
```

You will use this image for both client and server nodes in the system.

---

## **2. Docker Compose Setup for Multi-Node Deployment**

To deploy multiple nodes (clients and server) in the federated learning system, we use **Docker Compose**.

### **2.1 `docker-compose.yml`**

Here’s the **`docker-compose.yml`** file that defines the services for the **central server** and multiple **client nodes**:

```yaml
version: '3.8'
services:
  server_node:
    image: federated-node
    container_name: federated_server
    build: .
    ports:
      - "8000:8000"
    environment:
      - NODE_TYPE=server
    volumes:
      - ./server:/app
    networks:
      - federated_network
    command: ["python", "server_node.py"]

  client_node_1:
    image: federated-node
    container_name: federated_client_1
    build: .
    environment:
      - NODE_TYPE=client
      - CLIENT_ID=1
    volumes:
      - ./client_1:/app
    networks:
      - federated_network
    command: ["python", "client_node.py"]

  client_node_2:
    image: federated-node
    container_name: federated_client_2
    build: .
    environment:
      - NODE_TYPE=client
      - CLIENT_ID=2
    volumes:
      - ./client_2:/app
    networks:
      - federated_network
    command: ["python", "client_node.py"]

  client_node_3:
    image: federated-node
    container_name: federated_client_3
    build: .
    environment:
      - NODE_TYPE=client
      - CLIENT_ID=3
    volumes:
      - ./client_3:/app
    networks:
      - federated_network
    command: ["python", "client_node.py"]

networks:
  federated_network:
    driver: bridge
```

### **Explanation**:
- **Server Node**: Runs the **server_node.py** script, acting as the central aggregation point.
- **Client Nodes**: Each client node runs **client_node.py**, handling local training.
- **Networks**: All services communicate over the same **federated_network** bridge.

### **2.2 Starting the Federated System**

To start the entire system, use **Docker Compose**:

```bash
docker-compose up --build
```

This command builds the images and starts the **server** and **client nodes**.

---

## **3. Configuring Cloud Deployment (AWS EC2)**

If you plan to deploy the system on **AWS EC2**, follow these steps:

### **3.1 Setting Up AWS EC2 Instances**

1. Launch several EC2 instances (one for the server, multiple for clients).
2. Ensure that all instances are in the same **VPC** and can communicate with each other.
3. Open port **8000** in the **security groups** to allow communication.

### **3.2 SSH Into Each Instance**

Use the following command to SSH into each instance:

```bash
ssh -i "your-key.pem" ec2-user@ec2-xx-xxx-xx-xx.compute-1.amazonaws.com
```

### **3.3 Installing Docker on EC2 Instances**

Once inside the instance, install Docker:

```bash
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```

Log out and log back in to ensure the user has **Docker permissions**.

### **3.4 Cloning the Project**

Clone your federated learning project repository on each instance:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### **3.5 Running Docker Compose on EC2 Instances**

In the server instance:

```bash
docker-compose up --build server_node
```

In the client instances:

```bash
docker-compose up --build client_node_1
docker-compose up --build client_node_2
# And so on for each client node
```

---

## **4. Syft and TensorFlow Setup**

### **4.1 Syft Integration for Federated Learning**

PySyft allows secure and privacy-preserving federated learning. You will need to configure Syft in your **server_node.py** and **client_node.py** scripts.

In **client_node.py**, configure Syft for local training:

```python
import syft as sy
import tensorflow as tf

client = sy.VirtualMachineClient()

# Load the model and dataset
model = ...  # Your TensorFlow model
dataset = ...  # Your dataset

# Perform local training on the client
def local_training(model, dataset):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=5)
    return model

# Perform federated learning with Syft
local_model = local_training(model, dataset)
client.send(local_model)  # Send the updated model back to the server
```

In **server_node.py**, configure the server for model aggregation:

```python
import syft as sy

server = sy.VirtualMachine(name="server")

# Define the aggregation function (e.g., Federated Averaging)
def aggregate_models(models):
    avg_model = ...  # Implement your FedAvg logic here
    return avg_model

while True:
    client_models = server.get_models()  # Retrieve client models
    global_model = aggregate_models(client_models)
    server.send(global_model)  # Send the global model back to clients
```

---

## **5. Monitoring and Evaluation**

### **5.1 Monitoring System Performance**

To monitor the system’s resource usage (CPU, memory), you can use Docker’s monitoring tools or other cloud services:

```bash
docker stats
```

This command will display real-time resource usage for each container.

### **5.2 Evaluating the Global Model**

Once the federated training rounds are complete, you can evaluate the **global model** on a held-out test dataset.

In **server_node.py**:

```python
def evaluate_global_model(global_model, test_data):
    loss, accuracy = global_model.evaluate(test_data)
    print(f"Global Model Accuracy: {accuracy * 100:.2f}%")
```

---

## **Conclusion**

This deployment guide provides a comprehensive, step-by-step approach to setting up a federated learning system using **Docker**, **Syft**, and **TensorFlow**. The system is capable of scaling across multiple nodes, either locally or in the cloud, with minimal overhead. Proper configuration of **Docker Compose**, **Syft**, and **TensorFlow** ensures that the system can handle decentralized data and securely aggregate models from client nodes.
# **Model Architectures for Federated Learning**

## **Introduction**

This document provides a detailed overview of the model architectures used in our **federated learning** system. The models are designed to be compatible with **TensorFlow** and **PySyft** to enable decentralized training on client nodes. The document explains the key architectures used, such as **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks, and highlights how they fit into the federated learning setup.

### **Key Components:**

1. **CNN Architecture**: Used for image-based tasks such as image classification.
2. **LSTM Architecture**: Used for time-series or sequence-based data (e.g., text).
3. **Utility Functions**: Helper functions for model initialization, training, and evaluation.

---

## **1. Convolutional Neural Network (CNN)**

### **Overview**
A **Convolutional Neural Network (CNN)** is widely used for image-related tasks. It automatically learns to extract features from images through convolutional layers followed by pooling layers, making it ideal for image classification tasks like those in the **FEMNIST dataset**.

**Architecture Summary**:
- **Conv2D Layers**: For feature extraction through convolutional operations.
- **MaxPooling2D**: For reducing spatial dimensions and computational complexity.
- **Fully Connected Layers**: To classify the extracted features into predefined classes.

### **TensorFlow Implementation of CNN**

The following is the implementation of a **CNN model** using TensorFlow, with hooks for **PySyft** to enable federated learning. This architecture is especially effective for image classification tasks like **FEMNIST**.

```python
import tensorflow as tf
import syft as sy

# Hook TensorFlow with Syft
hook = sy.TFHook(tf)

def build_cnn_model(input_shape, num_classes):
    """Builds a Convolutional Neural Network (CNN) for image classification."""
    model = tf.keras.models.Sequential()

    # Convolutional Layer 1
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Flatten the output and add fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

# Compile the model
def compile_model(model):
    """Compiles the CNN model with optimizer and loss function."""
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

### **Key Layers**:
1. **Conv2D**: Convolutional layer that applies filters to the input image.
2. **MaxPooling2D**: Reduces the spatial dimensions of the feature map.
3. **Flatten**: Flattens the 2D output into a 1D vector.
4. **Dense**: Fully connected layers to classify the image into different categories.

### **Training in a Federated Setting**

Each client node trains the CNN locally on its dataset, and model updates (weights) are shared with the central server for aggregation:

```python
# Train the model locally on each client
model = build_cnn_model(input_shape=(28, 28, 1), num_classes=62)  # FEMNIST has 62 classes
model = compile_model(model)

# Train locally
model.fit(client_data, client_labels, epochs=5)

# Send local model to the central server for aggregation
model.send(server)
```

---

## **2. Long Short-Term Memory (LSTM)**

### **Overview**
**Long Short-Term Memory (LSTM)** networks are effective for sequential or time-series data. This architecture is commonly used for tasks like text classification or next-word prediction, where the model needs to learn dependencies across time steps.

**Architecture Summary**:
- **LSTM Layers**: Capture temporal dependencies in the sequence data.
- **Fully Connected Layers**: Used to map the LSTM output to the target labels.

### **TensorFlow Implementation of LSTM**

Below is the implementation of an **LSTM model** using TensorFlow for tasks like **next-word prediction**. The architecture is well-suited for sequential data and integrates with **PySyft** for federated learning.

```python
import tensorflow as tf
import syft as sy

# Hook TensorFlow with Syft
hook = sy.TFHook(tf)

def build_lstm_model(input_shape, num_classes):
    """Builds an LSTM model for sequence data (e.g., text)."""
    model = tf.keras.models.Sequential()

    # LSTM Layer
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=False))

    # Fully connected layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

# Compile the model
def compile_lstm_model(model):
    """Compiles the LSTM model with optimizer and loss function."""
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

### **Key Layers**:
1. **LSTM**: Recurrent layer that captures temporal patterns in sequence data.
2. **Dense**: Fully connected layer for classification.

### **Training in a Federated Setting**

LSTM models are trained on sequence data locally on client nodes. The model updates are securely aggregated at the server node:

```python
# Build and compile LSTM model
model = build_lstm_model(input_shape=(100, 50), num_classes=2)  # Example input shape for text
model = compile_lstm_model(model)

# Train the model locally
model.fit(client_sequences, client_labels, epochs=5)

# Send updates to the server
model.send(server)
```

---

## **3. Model Utility Functions**

### **Model Utils Overview**
This section provides helper functions for model management. These include:
- **Loading Pretrained Models**: For reusing previously trained models.
- **Saving Model Weights**: To store local model updates for federated aggregation.
- **Model Evaluation**: For evaluating model performance on test data.

### **Implementation of Model Utilities**

```python
import tensorflow as tf

def save_model_weights(model, file_path):
    """Save model weights to a specified file path."""
    model.save_weights(file_path)
    print(f"Model weights saved to {file_path}")

def load_model_weights(model, file_path):
    """Load model weights from a specified file path."""
    model.load_weights(file_path)
    print(f"Model weights loaded from {file_path}")

def evaluate_model(model, test_data, test_labels):
    """Evaluate the model on test data."""
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {accuracy}")
    return accuracy
```

### **Example Usage**:
- **Saving Model Weights**:

```python
save_model_weights(model, "path/to/model_weights.h5")
```

- **Loading Model Weights**:

```python
load_model_weights(model, "path/to/model_weights.h5")
```

---

## **Federated Learning with PySyft Integration**

In this architecture, both CNN and LSTM models can be used in a federated learning setting where the client nodes train the models locally, and the central server aggregates the updates.

### **Federated Learning Process with Model Architecture**:
1. **Client Node**:
   - Each client node trains its model (CNN for images, LSTM for sequences).
   - The model updates (weights) are securely sent to the server using **PySyft**.
   
2. **Server Node**:
   - The server aggregates the updates using a strategy like **FedAvg** and updates the global model.
   - The updated global model is sent back to the clients for the next training round.

### **Example Federated Training Workflow**:

```python
# Each client trains a CNN or LSTM locally
model = build_cnn_model(input_shape=(28, 28, 1), num_classes=62)
model.fit(client_data, client_labels, epochs=5)

# Send updates to the server
model.send(server)

# On the server side, aggregate the updates using FedAvg
aggregated_model = federated_averaging(global_model, client_models)

# Send the updated global model back to the clients
aggregated_model.send(clients)
```

---

## **Conclusion**

This document provided an overview of the various model architectures used in our **federated learning system** built with **TensorFlow** and **PySyft**. The architectures, including CNN and LSTM, were discussed in detail, with examples of how they can be used in both local and federated
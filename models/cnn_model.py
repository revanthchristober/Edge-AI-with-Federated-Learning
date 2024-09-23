import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler

class AdvancedCNNModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(AdvancedCNNModel, self).__init__()

        # Block 1
        self.conv1_1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=input_shape)
        self.bn1_1 = layers.BatchNormalization()
        self.conv1_2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn1_2 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))

        # Block 2
        self.conv2_1 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn2_1 = layers.BatchNormalization()
        self.conv2_2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn2_2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))

        # Block 3
        self.conv3_1 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn3_1 = layers.BatchNormalization()
        self.conv3_2 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn3_2 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))

        # Block 4 - Residual Block
        self.conv4_1 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn4_1 = layers.BatchNormalization()
        self.conv4_2 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn4_2 = layers.BatchNormalization()
        self.residual_block = layers.Conv2D(512, (1, 1), padding='same')

        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2))

        # Fully Connected Layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Block 1
        x = self.conv1_1(inputs)
        x = self.bn1_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.bn2_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3_1(x)
        x = self.bn3_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool3(x)

        # Block 4 with residual connection
        residual = self.residual_block(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x, training=training)
        x = tf.nn.relu(x)

        x = layers.add([x, residual])
        x = self.pool4(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout1(x)
        x = self.fc2(x)
        if training:
            x = self.dropout2(x)
        return self.fc3(x)

def create_advanced_cnn_model(input_shape, num_classes):
    """Utility function to create and compile the advanced CNN model."""
    model = AdvancedCNNModel(input_shape, num_classes)
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    """Learning rate scheduling based on the epoch number."""
    if epoch < 10:
        return lr
    elif epoch < 30:
        return lr * 0.1
    else:
        return lr * 0.01

def get_lr_scheduler():
    return LearningRateScheduler(lr_scheduler)

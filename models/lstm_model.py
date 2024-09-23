import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

class AdvancedLSTMModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes, bidirectional=True, attention=False):
        super(AdvancedLSTMModel, self).__init__()

        # Embedding layer
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)

        # LSTM Layer
        self.lstm_units = lstm_units
        if bidirectional:
            self.lstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        else:
            self.lstm = layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)

        # Attention Mechanism (Optional)
        self.attention = attention
        if self.attention:
            self.attention_layer = layers.Attention()

        # Fully Connected Layers
        self.fc1 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dropout2 = layers.Dropout(0.5)

        # Output Layer
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Embedding layer
        x = self.embedding(inputs)

        # LSTM layer (bidirectional if set)
        x = self.lstm(x)

        # Attention mechanism
        if self.attention:
            x = self.attention_layer([x, x])

        # Global Max Pooling after LSTM
        x = layers.GlobalMaxPooling1D()(x)

        # Fully connected layers
        x = self.fc1(x)
        if training:
            x = self.dropout1(x)
        x = self.fc2(x)
        if training:
            x = self.dropout2(x)

        # Output layer
        return self.output_layer(x)

# Utility function to create the model
def create_advanced_lstm_model(vocab_size, embedding_dim, lstm_units, num_classes, bidirectional=True, attention=False):
    """Utility function to create and compile the advanced LSTM model."""
    model = AdvancedLSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, bidirectional, attention)
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

# Gradient Clipping Function for training
def clip_gradients(optimizer, clip_value):
    """Custom gradient clipping."""
    gradients = optimizer.get_gradients(optimizer.total_loss, model.trainable_weights)
    clipped_gradients = [tf.clip_by_value(grad, -clip_value, clip_value) for grad in gradients]
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_weights))

# Custom Callback for Gradient Clipping
class GradientClippingCallback(tf.keras.callbacks.Callback):
    def __init__(self, clip_value):
        super(GradientClippingCallback, self).__init__()
        self.clip_value = clip_value

    def on_train_batch_end(self, batch, logs=None):
        clip_gradients(self.model.optimizer, self.clip_value)

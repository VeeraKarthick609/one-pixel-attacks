import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import numpy as np


class SimpleCNN:
    def __init__(self, epochs=50, batch_size=64, load_weights = True):
        self.name = 'simple_cnn'
        self.model_filename = 'networks/model/simple_cnn_model.keras'
        self.epochs = epochs
        self.batch_size = batch_size

        # Load and preprocess CIFAR-10 dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar10.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        self.datagen.fit(self.x_train)

        # Build the CNN model
        self.model = self.build_model()
    
    def count_params(self):
        return self.model.count_params()

    def build_model(self):
        model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Conv Block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Conv Block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense Layers
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self):
        # Train the model
        history = self.model.fit(self.datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                                 epochs=self.epochs,
                                 validation_data=(self.x_test, self.y_test))
        # Save model
        self.model.save(self.model_filename)
        return history

    def evaluate(self):
        # Evaluate the model
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print(f"\nTest accuracy: {test_acc}")
        return test_loss, test_acc
    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        processed = self.color_process(img)
        return self.model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self.model.evaluate(x_test, y_test, verbose=0)[1]
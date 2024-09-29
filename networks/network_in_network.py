import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.initializers import RandomNormal  
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

from networks.train_plot import PlotLearning

class NetworkInNetwork:
    def __init__(self, epochs=200, batch_size=128, load_weights=True):
        self.name               = 'net_in_net'
        self.model_filename     = 'networks/models/net_in_net.h5'
        self.num_classes        = 10
        self.input_shape        = 32, 32, 3
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = 391
        self.weight_decay       = 0.0001
        self.dropout            = 0.5
        self.log_filepath       = r'networks/models/net_in_net/'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)
    
    def count_params(self):
        return self._model.count_params()

    def color_preprocessing(self, x_train, x_test):
        mean = np.array([125.307, 122.95, 113.865])
        std = np.array([62.9932, 62.0887, 66.7048])
        x_train = (x_train.astype('float32') - mean) / std
        x_test = (x_test.astype('float32') - mean) / std
        return x_train, x_test

    def scheduler(self, epoch):
        if epoch <= 60:
            return 0.05
        if epoch <= 120:
            return 0.01
        if epoch <= 160:    
            return 0.002
        return 0.0004

    def build_model(self):
        model = Sequential([
            Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal", input_shape=self.input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'),
            Dropout(self.dropout),
            
            Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'),
            Dropout(self.dropout),
            
            Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay), kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu'),
            
            GlobalAveragePooling2D(),
            Activation('softmax')
        ])
        
        sgd = optimizers.SGD(learning_rate=.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def train(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        model = self.build_model()
        model.summary()

        checkpoint = ModelCheckpoint(self.model_filename,
                                    monitor='val_loss', 
                                    save_best_only=True,
                                    mode='auto')
        plot_callback = PlotLearning()
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        lr_scheduler = LearningRateScheduler(self.scheduler)

        cbks = [checkpoint, plot_callback, tb_cb, lr_scheduler]

        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.125,
            height_shift_range=0.125,
            fill_mode='constant',
            cval=0.
        )
        datagen.fit(x_train)

        model.fit(
            datagen.flow(x_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=self.iterations,
            epochs=self.epochs,
            callbacks=cbks,
            validation_data=(x_test, y_test)
        )
        
        model.save(self.model_filename)
        self._model = model

    def color_process(self, imgs):
        imgs = np.array(imgs, dtype='float32')
        mean = np.array([125.307, 122.95, 113.865])
        std = np.array([62.9932, 62.0887, 66.7048])
        return (imgs - mean) / std

    def predict(self, img):
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (_, _), (x_test, y_test) = cifar10.load_data()
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        x_test = self.color_process(x_test)
        return self._model.evaluate(x_test, y_test, verbose=0)[1]
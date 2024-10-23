import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from networks.train_plot import PlotLearning  # assuming this is your custom callback

# Uncomment this if you want to use Differential Privacy
# import tensorflow_privacy
# from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

class ResNetMasked:
    def __init__(self, epochs=200, batch_size=128, load_weights=True, use_dp=False):
        self.name = 'resnet_masked'
        self.model_filename = 'networks/models/resnet_masked.keras'
        
        self.stack_n = 5    
        self.num_classes = 10
        self.img_rows, self.img_cols = 32, 32
        self.img_channels = 3
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = 50000 // self.batch_size
        self.weight_decay = 0.0001
        self.log_filepath = r'networks/models/resnet_masked/'
        self.use_dp = use_dp  # Use Differential Privacy or not

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)

    def count_params(self):
        return self._model.count_params()

    def color_preprocessing(self, x_train, x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        return x_train, x_test

    def scheduler(self, epoch):
        if epoch < 80:
            return 0.1
        if epoch < 150:
            return 0.01
        return 0.001

    def residual_network(self, img_input, classes_num=10, stack_n=5):
        def residual_block(input, out_channel, increase=False):
            stride = (2, 2) if increase else (1, 1)

            pre_bn = BatchNormalization()(input)
            pre_relu = Activation('relu')(pre_bn)

            conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(pre_relu)
            bn_1 = BatchNormalization()(conv_1)
            relu1 = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(relu1)
            if increase:
                projection = Conv2D(out_channel,
                                    kernel_size=(1, 1),
                                    strides=(2, 2),
                                    padding='same',
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=regularizers.l2(self.weight_decay))(input)
                block = add([conv_2, projection])
            else:
                block = add([input, conv_2])
            return block

        # Building the model
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

        for _ in range(stack_n):
            x = residual_block(x, 16, False)

        x = residual_block(x, 32, True)
        for _ in range(1, stack_n):
            x = residual_block(x, 32, False)
        
        x = residual_block(x, 64, True)
        for _ in range(1, stack_n):
            x = residual_block(x, 64, False)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(classes_num, activation='softmax',
                  kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        return x

    def train(self):
        # Load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        # Build network
        img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels))
        output = self.residual_network(img_input, self.num_classes, self.stack_n)
        resnet = Model(img_input, output)
        resnet.summary()

        # Gradient Clipping Option (default)
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True, clipnorm=1.0)
        
        # Uncomment if you want to use Differential Privacy instead
        # dp_sgd = DPKerasSGDOptimizer(
        #     l2_norm_clip=1.0,  # Clip gradients
        #     noise_multiplier=1.1,  # Add noise to gradients
        #     num_microbatches=self.batch_size,
        #     learning_rate=0.1
        # )
        # optimizer = dp_sgd if self.use_dp else sgd
        optimizer = sgd  # Default to Gradient Clipping

        resnet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Set callbacks
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(self.scheduler)
        checkpoint = ModelCheckpoint(self.model_filename, 
                                     monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        plot_callback = PlotLearning()  # Custom callback
        cbks = [change_lr, tb_cb, checkpoint, plot_callback]

        # Set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     width_shift_range=0.125,
                                     height_shift_range=0.125,
                                     fill_mode='constant', cval=0.)
        datagen.fit(x_train)

        # Start training
        resnet.fit(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                   steps_per_epoch=self.iterations,
                   epochs=self.epochs,
                   callbacks=cbks,
                   validation_data=(x_test, y_test))
        resnet.save(self.model_filename)

        self._model = resnet
        self.param_count = self._model.count_params()

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
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._model.evaluate(x_test, y_test, verbose=0)[1]

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import models
from tensorflow.keras import optimizers
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Any, Tuple

from config import IMG_SIZE


class CNNModel:

    num_dense_layers = 2
    dense_neurons = [128, 64]
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    kernel_size = (3, 3)
    filters_per_layer = 128

    def __init__(self, num_conv_layers: int) -> None:
        self.model = models.Sequential()
        self.num_conv_layers = num_conv_layers
        self.build_model()

    def build_model(self) -> None:
        for i in range(self.num_conv_layers):
            if i == 0:
                self.model.add(Conv2D(self.filters_per_layer, self.kernel_size, activation='relu',
                                      input_shape=self.input_shape))
            else:
                self.model.add(Conv2D(self.filters_per_layer, self.kernel_size, activation='relu'))
                self.model.add(Dropout(0.1))
            self.model.add(BatchNormalization())
            self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))

        self.model.add(GlobalAveragePooling2D())

        for j in range(self.num_dense_layers - 1):
            self.model.add(Dense(self.dense_neurons[j], activation='relu'))
            self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.summary()

    def compile_model(self, optimizer_name: str, learning_rate: float) -> None:
        optimizer = self.get_optimizer(optimizer_name, learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def fit_model(self, train_data: Any, val_data: Any, epochs: int) -> Any:
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[early_stopping, reduce_lr]
        )
        return history

    @staticmethod
    def get_optimizer(name: str, learning_rate: float) -> optimizers.Optimizer:
        optimizers_dict = {
            'adam': optimizers.Adam(learning_rate=learning_rate),
            'rmsprop': optimizers.RMSprop(learning_rate=learning_rate),
            'sgd': optimizers.SGD(learning_rate=learning_rate)
        }
        return optimizers_dict.get(name.lower(), optimizers.Adam(learning_rate=learning_rate))

    def evaluate_model(self, test_data: Any) -> Tuple[float, float]:
        test_loss, test_acc = self.model.evaluate(test_data)
        return test_loss, test_acc

    def predict(self, data: Any) -> Any:
        return self.model.predict(data)

    def get_model(self) -> models.Model:
        return self.model

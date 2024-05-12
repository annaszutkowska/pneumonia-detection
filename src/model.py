import os
from typing import Tuple, List

from config import IMG_SIZE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import models


class CNNModel:

    def __init__(self, num_conv_layers: int, filters_per_layer: List[int], kernel_size: Tuple[int, int],
                 pool_size: Tuple[int, int], num_dense_layers: int, dense_neurons: List[int], dropout_rate: float) \
            -> None:
        self.model = models.Sequential()
        self.input_shape = (IMG_SIZE, IMG_SIZE, 1)
        self.num_classes = 2
        self.num_conv_layers = num_conv_layers
        self.filters_per_layer = filters_per_layer
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_dense_layers = num_dense_layers
        self.dense_neurons = dense_neurons
        self.dropout_rate = dropout_rate
        self.build_model()

    def build_model(self) -> None:
        for i in range(self.num_conv_layers):
            if i == 0:
                self.model.add(Conv2D(self.filters_per_layer[i], self.kernel_size, activation='relu',
                                      input_shape=self.input_shape))
            else:
                self.model.add(Conv2D(self.filters_per_layer[i], self.kernel_size, activation='relu'))
            self.model.add(MaxPooling2D(self.pool_size))

        self.model.add(Flatten())

        for j in range(self.num_dense_layers):
            self.model.add(Dense(self.dense_neurons[j], activation='relu'))
            if self.dropout_rate > 0:
                self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.num_classes, activation='sigmoid'))

    def get_model(self) -> models.Model:
        return self.model

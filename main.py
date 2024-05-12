import os
from typing import Any

import numpy as np
from keras.src.utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, DATASET_LABELS, LABELS_DICT
from src.data_processing import get_training_data, shuffle_and_split_data
from src.data_visualization import plot_dataset_proportions, plot_class_distribution, show_sample_images, \
    plot_confusion_matrix, plot_training_history
from src.model import CNNModel

num_conv_layers = 3
filters_per_layer = [32, 64, 128]
kernel_size = (3, 3)
pool_size = (2, 2)
num_dense_layers = 2
dense_neurons = [128, 64]
dropout_rate = 0.5
optimizer = "Adam"
epochs = 30


def get_input(prompt: str, default: Any) -> str:
    user_input = input(prompt + f" (default {default}): ")
    return user_input.strip() or default


if __name__ == "__main__":
    print("Loading the data")
    train_images, train_labels = get_training_data(TRAIN_DIR, LABELS_DICT)
    train_images, train_labels = train_images[:500], train_labels[:500]
    val_images, val_labels = get_training_data(VAL_DIR, LABELS_DICT)
    test_images, test_labels = get_training_data(TEST_DIR, LABELS_DICT)

    print("Visualizing the dataset")
    dataset_sizes = [len(train_images), len(val_images), len(test_images)]
    plot_dataset_proportions(dataset_sizes, DATASET_LABELS)
    plot_class_distribution(train_labels)
    show_sample_images(train_images, train_labels)

    print("Correcting data imbalance")
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = shuffle_and_split_data(
        np.concatenate([train_images, val_images, test_images]),
        np.concatenate([train_labels, val_labels, test_labels])
    )
    print(f"New dataset sizes: train - {len(train_images)}, val - {len(val_images)}, test - {len(test_images)}")

    print("Visualizing the dataset after correcting the imbalance")
    dataset_sizes = [len(train_images), len(val_images), len(test_images)]
    plot_dataset_proportions(dataset_sizes, DATASET_LABELS)
    plot_class_distribution(train_labels)

    print("Loading the model")
    num_conv_layers = int(get_input("Enter number of convolutional layers", num_conv_layers))
    filters_per_layer_input = get_input("Enter filters per layer separated by comma",
                                        ','.join(map(str, filters_per_layer)))
    filters_per_layer = list(map(int, filters_per_layer_input.split(',')))
    kernel_size = tuple(map(int, get_input("Enter kernel size as two numbers separated by comma",
                                           f"{kernel_size[0]},{kernel_size[1]}").split(',')))
    pool_size = tuple(map(int, get_input("Enter pool size as two numbers separated by comma",
                                         f"{pool_size[0]},{pool_size[1]}").split(',')))
    num_dense_layers = int(get_input("Enter number of dense layers", num_dense_layers))
    dense_neurons_input = get_input("Enter neurons per dense layer separated by comma",
                                    ','.join(map(str, dense_neurons)))
    dense_neurons = list(map(int, dense_neurons_input.split(',')))
    dropout_rate = float(get_input("Enter dropout rate", dropout_rate))

    cnn_model = CNNModel(num_conv_layers, filters_per_layer, kernel_size, pool_size, num_dense_layers, dense_neurons,
                         dropout_rate)
    model = cnn_model.get_model()
    model.summary()

    print("Compiling")

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_labels = to_categorical(train_labels, num_classes=2)
    val_labels = to_categorical(val_labels, num_classes=2)
    test_labels = to_categorical(test_labels, num_classes=2)

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    validation_generator = val_datagen.flow(val_images, val_labels, batch_size=32)
    test_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

    optimizer = get_input("Enter optimizer", optimizer)
    epochs = int(get_input("Enter number of epochs", epochs))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[ModelCheckpoint("best_model.keras", save_best_only=True),
                   EarlyStopping(patience=10, restore_best_weights=True)])

    print("Testing and results")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

    predictions = model.predict(test_generator)
    predictions = predictions.flatten()
    predictions = np.where(predictions > 0.5, 1, 0)
    y_true = test_labels

    class_names = np.array(['Normal', 'Pneumonia'])
    plot_confusion_matrix(y_true, predictions, classes=class_names, normalize=True)
    plot_training_history(history)

    print("Done")

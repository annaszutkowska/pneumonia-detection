import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import Any
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from src.data_processing import get_training_data, shuffle_and_split_data
from src.data_visualization import plot_confusion_matrix, plot_training_history, plot_dataset_proportions, \
    plot_class_distribution, show_sample_images

from src.model import CNNModel
from config import TRAIN_DIR, LABELS_DICT, VAL_DIR, TEST_DIR, DATASET_LABELS


def get_input(prompt: str, default: Any) -> Any:
    response = input(f"{prompt} (default={default}): ")
    return response if response else default


if __name__ == '__main__':
    print("Getting and processing data")
    train_images, train_labels = get_training_data(TRAIN_DIR, LABELS_DICT)
    val_images, val_labels = get_training_data(VAL_DIR, LABELS_DICT)
    test_images, test_labels = get_training_data(TEST_DIR, LABELS_DICT)

    print("Visualizing the dataset")
    dataset_sizes = [len(train_images), len(val_images), len(test_images)]
    plot_dataset_proportions(dataset_sizes, DATASET_LABELS)
    plot_class_distribution(train_labels)
    show_sample_images(train_images, train_labels)

    print("Correcting data imbalance")
    (train_images, train_labels), (val_images, val_labels) = shuffle_and_split_data(
        np.concatenate([train_images, val_images]),
        np.concatenate([train_labels, val_labels])
    )
    print(f"New dataset sizes: train - {len(train_images)}, val - {len(val_images)}, test - {len(test_images)}")

    print("Visualizing the dataset after correcting the imbalance")
    dataset_sizes = [len(train_images), len(val_images), len(test_images)]
    plot_dataset_proportions(dataset_sizes, DATASET_LABELS)
    plot_class_distribution(train_labels)

    print("Initializing data generators")
    datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    datagen.fit(train_images)

    train_generator = datagen.flow(train_images, train_labels)
    validation_generator = datagen.flow(val_images, val_labels)

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(test_images, test_labels)

    optimizer_name = get_input("Enter optimizer", "adam")
    learning_rate = float(
        get_input("Enter learning rate, skip for optimizer's default", 0.01 if optimizer_name == 'sgd' else 0.001))
    num_conv_layers = int(get_input("Enter number of conv layers", 3))
    cnn_model = CNNModel(num_conv_layers)
    cnn_model.compile_model(optimizer_name, learning_rate)

    epochs = int(get_input("Enter number of epochs", 10))

    print("Training the model")
    history = cnn_model.fit_model(train_generator, validation_generator, epochs)

    print("Testing and results")
    test_loss, test_acc = cnn_model.evaluate_model(test_generator)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

    predictions = cnn_model.predict(test_images)
    predictions = np.where(predictions > 0.5, 1, 0).flatten()

    print("Predictions:", predictions[:20])
    print("True Labels:", test_labels[:20])

    class_names = np.array(['Normal', 'Pneumonia'])
    plot_confusion_matrix(test_labels, predictions, classes=class_names, normalize=True)
    plot_training_history(history)

    print("Done")

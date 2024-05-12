import os
from typing import Tuple, Dict

import cv2
import numpy as np
from numpy import ndarray


def shuffle_data(*datasets: ndarray) -> ndarray:
    combined_data = np.concatenate(datasets, axis=0)
    np.random.shuffle(combined_data)
    return combined_data


def shuffle_and_split_data(images: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8,
                           validation_ratio: float = 0.1) -> \
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    data = np.column_stack((images.reshape(images.shape[0], -1), labels))

    np.random.shuffle(data)

    total_samples = data.shape[0]
    train_end = int(total_samples * train_ratio)
    validation_end = train_end + int(total_samples * validation_ratio)

    train = data[:train_end]
    validation = data[train_end:validation_end]
    test = data[validation_end:]

    train_images, train_labels = train[:, :-1], train[:, -1]
    validation_images, validation_labels = validation[:, :-1], validation[:, -1]
    test_images, test_labels = test[:, :-1], test[:, -1]

    image_shape = images.shape[1:]
    train_images = train_images.reshape(-1, *image_shape)
    validation_images = validation_images.reshape(-1, *image_shape)
    test_images = test_images.reshape(-1, *image_shape)

    return (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels)


def get_training_data(directory: str, labels_dict: Dict[str, int], target_size: Tuple[int, int] = (150, 150)) -> (
        Tuple)[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for label, label_idx in labels_dict.items():
        label_dir = os.path.join(directory, label)
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            img = preprocess_image(file_path)
            images.append(img)
            labels.append(label_idx)
    images = np.array(images).reshape(-1, target_size[0], target_size[1], 1)
    labels = np.array(labels)
    return images, labels


def preprocess_image(img_path: str) -> ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    return img

import os

import cv2
import numpy as np

from config import IMG_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, DATASET_LABELS, CLASS_LABELS
from src.data_visualization import plot_dataset_proportions, plot_class_distribution, show_sample_images


def get_training_data(data_dir: str) -> np.ndarray:
    data = []
    for label in CLASS_LABELS:
        path = os.path.join(data_dir, label)
        class_num = CLASS_LABELS.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)


if __name__ == "__main__":
    print("Loading the data")
    train, val, test = (get_training_data(DIR) for DIR in [TRAIN_DIR, VAL_DIR, TEST_DIR])

    print("Visualizing the dataset")
    dataset_sizes = [len(train), len(val), len(test)]
    plot_dataset_proportions(dataset_sizes, DATASET_LABELS)
    plot_class_distribution(train)
    show_sample_images(train, CLASS_LABELS)

    print("Correcting data imbalance")

    # Load the model

    # Load and preprocess the test data

    # Evaluate the model on the test data

    # Print evaluation results
    print("Done")

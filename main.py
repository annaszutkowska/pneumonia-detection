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


def shuffle_data(*args: np.ndarray) -> np.ndarray:
    combined_data = np.concatenate(args, axis=0)
    np.random.shuffle(combined_data)
    return combined_data


if __name__ == "__main__":
    print("Loading the data")
    train, val, test = (get_training_data(DIR) for DIR in [TRAIN_DIR, VAL_DIR, TEST_DIR])

    print("Visualizing the dataset")
    dataset_sizes = [len(train), len(val), len(test)]
    plot_dataset_proportions(dataset_sizes, DATASET_LABELS)
    plot_class_distribution(train)
    show_sample_images(train, CLASS_LABELS)

    print("Correcting data imbalance")
    combined_dataset = shuffle_data(train, val, test)
    num_train = int(0.8 * len(combined_dataset))
    num_val = int(0.1 * len(combined_dataset))
    num_test = len(combined_dataset) - num_train - num_val
    train = combined_dataset[:num_train]
    val = combined_dataset[num_train:num_train + num_val]
    test = combined_dataset[num_train + num_val:]
    print(f"New dataset sizes: train - {len(train)}, val - {len(val)}, test - {len(test)}")

    print("Visualizing the dataset after correcting the imbalance")
    dataset_sizes = [len(train), len(val), len(test)]
    plot_dataset_proportions(dataset_sizes, DATASET_LABELS)
    plot_class_distribution(train)

    # Load the model

    # Load and preprocess the test data

    # Evaluate the model on the test data

    # Print evaluation results
    print("Done")

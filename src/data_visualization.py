from typing import List, Dict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from config import CLASS_LABELS, LABELS_DICT


def plot_dataset_proportions(dataset_sizes: List[int], dataset_labels: List[str]) -> None:
    total = sum(dataset_sizes)
    proportions = [size / total * 100 for size in dataset_sizes]
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=dataset_labels, y=proportions, hue=dataset_labels, palette='pastel', dodge=False)
    plt.title('Dataset Proportions')
    plt.xlabel('Dataset')
    plt.ylabel('Percentage of Total (%)')
    for i, prop in enumerate(proportions):
        ax.text(i, prop, f'{prop:.2f}%', ha='center', va='bottom', fontweight='bold', color='teal')
    plt.show()


def plot_class_distribution(labels: np.ndarray) -> None:
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    if labels.dtype == 'int':
        unique_labels = np.unique(labels)
        label_names = {label: f"Class {label}" for label in unique_labels}
        labels = np.array([label_names[label] for label in labels])

    sns.histplot(labels, discrete=True, color='lightpink', edgecolor='black', linewidth=1.5)
    plt.title('Distribution of Classes in Training Set', fontsize=14, fontweight='bold', color='teal')
    plt.xlabel('Class Type', fontsize=12, fontweight='bold', color='cadetblue')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold', color='cadetblue')
    plt.xticks(rotation=45, fontsize=10, fontweight='bold',
               color='palevioletred')
    plt.yticks(fontsize=10, fontweight='bold', color='palevioletred')
    plt.show()


def show_sample_images(images: np.ndarray, labels: np.ndarray) -> None:
    fig, axes = plt.subplots(1, len(CLASS_LABELS), figsize=(5 * len(CLASS_LABELS), 5))
    for ax, class_label in zip(axes, CLASS_LABELS):
        label_index = LABELS_DICT[class_label]
        for i, label in enumerate(labels):
            if label == label_index:
                ax.imshow(images[i], cmap='gray')
                ax.set_title(f'{class_label} Image', color='teal')
                ax.axis('off')
                break
        else:
            ax.set_title(f'No {class_label} Image Found', color='red')
            ax.axis('off')

    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray,
                          normalize: bool = False, cmap: plt.cm = plt.cm.Blues) -> None:
    title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[np.unique(y_true)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_training_history(history: Dict[str, List[float]]) -> None:
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'r*-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

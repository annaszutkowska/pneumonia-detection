from typing import List

import seaborn as sns
from matplotlib import pyplot as plt


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


def plot_class_distribution(train_dataset: np.ndarray) -> None:
    labels_list = ["Pneumonia" if i[1] == 0 else "Normal" for i in train_dataset]
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(labels_list, discrete=True, color='lightpink', edgecolor='black', linewidth=1.5)
    plt.title('Distribution of Classes in Training Set', fontsize=14, fontweight='bold', color='teal')
    plt.xlabel('Class Type', fontsize=12, fontweight='bold', color='cadetblue')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold', color='cadetblue')
    plt.xticks(fontsize=10, fontweight='bold', color='palevioletred')
    plt.yticks(fontsize=10, fontweight='bold', color='palevioletred')
    plt.show()


def show_sample_images(train_dataset: np.ndarray, labels: List[str]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(train_dataset[0][0], cmap='gray')
    axes[0].set_title(f'First Image: {labels[train_dataset[0][1]]}', color='palevioletred')
    axes[0].axis('off')
    axes[1].imshow(train_dataset[-1][0], cmap='gray')
    axes[1].set_title(f'Last Image: {labels[train_dataset[-1][1]]}', color='teal')
    axes[1].axis('off')
    plt.show()

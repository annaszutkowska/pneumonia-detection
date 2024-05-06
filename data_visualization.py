import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from config import LABELS, IMG_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR


def get_training_data(data_dir: str) -> np.ndarray:
    data = []
    for label in LABELS:
        path = os.path.join(data_dir, label)
        class_num = LABELS.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)


train, val, test = (get_training_data(DIR) for DIR in [TRAIN_DIR, VAL_DIR, TEST_DIR])

dataset_sizes = [len(dataset) for dataset in [train, val, test]]
dataset_labels = ['Train', 'Validation', 'Test']

total = sum(dataset_sizes)
proportions = [s / total * 100 for s in dataset_sizes]

plt.figure(figsize=(8, 4))
ax = sns.barplot(x=dataset_labels, y=proportions, palette='pastel')
plt.title('Dataset Proportions')
plt.xlabel('Dataset')
plt.ylabel('Percentage of Total (%)')

for i, prop in enumerate(proportions):
    ax.text(i, prop, f'{prop:.2f}%', ha='center', va='bottom', fontweight='bold', color='teal')
plt.show()

labels_list = ["Pneumonia" if i[1] == 0 else "Normal" for i in train]

sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.histplot(labels_list, discrete=True, color='lightpink', edgecolor='black', linewidth=1.5)
plt.title('Distribution of Classes in Training Set', fontsize=14, fontweight='bold', color='teal')
plt.xlabel('Class Type', fontsize=12, fontweight='bold', color='cadetblue')
plt.ylabel('Frequency', fontsize=12, fontweight='bold', color='cadetblue')
plt.xticks(fontsize=10, fontweight='bold', color='palevioletred')
plt.yticks(fontsize=10, fontweight='bold', color='palevioletred')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(train[0][0], cmap='gray')
axes[0].set_title(f'First Image: {LABELS[train[0][1]]}', color='palevioletred')
axes[0].axis('off')

axes[1].imshow(train[-1][0], cmap='gray')
axes[1].set_title(f'Last Image: {LABELS[train[-1][1]]}', color='teal')
axes[1].axis('off')

plt.show()

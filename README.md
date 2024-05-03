# Pneumonia Detection

## Project Overview
This project is a Convolutional Neural Network (CNN) based pneumonia detector.
It is designed to identify pneumonia from chest X-ray images using image processing and deep learning techniques.

## Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.12 or higher
- pip (Python package installer)

### Installation
Clone the repository to your local machine:

```
git clone https://github.com/<yourusername>/pneumonia-detector.git
cd pneumonia-detector
```

Install the required Python libraries:

```
pip install -r requirements.txt
```

### Downloading the Dataset
The dataset used in this project is hosted on Kaggle. To set up the dataset:

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (you will need a Kaggle account).
2. Download the dataset to your local machine.
3. Unzip the dataset and place it into the `./data` directory in the project folder.

You should see the dataset directories like train, test, and val.

### Running the Model
To run the model for testing, execute the following command:

```
python ./main.py
```


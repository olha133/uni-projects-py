# Cancer Cell Classification Using Neural Networks

## Project Overview

This project focuses on classifying cancer cells as benign or malignant using the Cancer Data dataset. The dataset contains 570 samples with 30 features (symptoms) that are used to determine the type of cancer.

### Key Features:
1. **Dataset:** Cancer Data, including 2 classes: benign (B) and malignant (M).
2. **Models:** Developed and tested 5 different neural network models to identify the most effective architecture.
3. **Optimization:** The best model was further evaluated using different optimizers, learning rates, and activation functions.
4. **Visualization:** Data distribution and feature relationships were visualized using `matplotlib` and `seaborn`.

---

## Dataset Preparation

The dataset is preprocessed as follows:
- Categorical labels (`B` and `M`) were encoded as `0` and `1`.
- Feature values were scaled to a normalized range using `MinMaxScaler`.
- Data was split into training and validation sets (80/20 split).

---

## Neural Network Models

Five models with varying architectures were tested. All models consist of fully connected layers with `ReLU` activation functions, and `BCEWithLogitsLoss` is used as the loss function.

### Example Architecture:
- Input Layer: Size based on the dataset features (30).
- Hidden Layers: Varying configurations (e.g., 64, 32, 16 nodes).
- Output Layer: Single node for binary classification.

---

## Results

1. The models were evaluated based on training and validation accuracy.
2. The best model was identified and fine-tuned using:
   - Different optimizers (e.g., Adam, SGD).
   - Learning rate variations.
   - Different activation functions.

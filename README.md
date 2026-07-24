# Breast Cancer Classification — K-Nearest Neighbors (KNN)

## Objective
A healthcare organization wants to develop a machine learning model to predict whether a breast tumor is Malignant (M) or Benign (B) based on diagnostic measurements. This project builds a K-Nearest Neighbors (KNN) classification model to classify tumors accurately.

## Dataset
Breast Cancer Wisconsin Diagnostic Dataset (Kaggle):
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

> Note: The dataset is **not** included in this repository. Download it from the Kaggle link above (file is typically named `data.csv`) and place it in the project root before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn (`sklearn`)

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, identified numerical diagnostic features and the target variable (`diagnosis`), and reviewed dataset info and summary statistics.
2. **Data Preprocessing** — Checked for missing values, dropped unnecessary columns (`id` and the empty `Unnamed: 32` artifact column), encoded the target variable (M → 1, B → 0), standardized feature values with `StandardScaler`, and split the data into 80% training / 20% testing sets.
3. **Model Development** — Trained a K-Nearest Neighbors classifier (`scikit-learn`) with K = 5 and predicted class labels for the test dataset.
4. **Model Evaluation** — Evaluated the model using Accuracy, Precision, Recall, and F1-Score, and generated a Confusion Matrix.
5. **Conclusion** — Summarized key findings, the importance of feature scaling in KNN, and a limitation of the KNN algorithm.

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 0.9561 |
| Precision | 0.9744 |
| Recall | 0.9048 |
| F1-Score | 0.9383 |

**Key finding:** The K=5 KNN model correctly classifies about 96% of tumors overall. Precision is higher than recall, meaning the model rarely raises a false alarm for Malignant cases, but it still misses a small number of actual Malignant cases — the more clinically concerning type of error.

## Conclusion
This project used a K-Nearest Neighbors (K=5) classifier to predict whether a breast tumor is Malignant or Benign based on diagnostic measurements. After preprocessing the data (removing the non-predictive `id` column and the empty `Unnamed: 32` artifact column, encoding the diagnosis label, and standardizing all features), the model achieved an accuracy of 0.9561, precision of 0.9744, recall of 0.9048, and an F1-score of 0.9383 on the test set — a strong result overall, though recall being somewhat lower than precision indicates a small number of Malignant cases were missed.

Feature scaling was essential for this model, since KNN classifies new points based on the distance to their nearest neighbors. Without standardization, features with naturally larger numeric ranges, such as `area_mean` (which reaches into the thousands), would dominate the distance calculation over features with smaller ranges, such as `smoothness_mean` (values under 1), regardless of each feature's actual diagnostic importance.

A key limitation of KNN is that it must compute the distance to every training point for each new prediction, making it computationally expensive as the dataset grows, and it treats all features as equally important unless explicitly weighted, making it sensitive to irrelevant or noisy features.

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook Assignment-4.ipynb
```

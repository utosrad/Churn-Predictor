# Telco Customer Churn Prediction using Deep Learning 

This project focuses on predicting customer churn for a telecommunications company. By analyzing customer data, we build a deep learning model to identify customers who are likely to cancel their service. This allows the company to proactively offer incentives and improve customer retention.

## Table of Contents
* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Workflow](#workflow)
* [Model Architecture](#model-architecture)
* [Results](#results)
* [Requirements](#requirements)
* [How to Run](#how-to-run)

---

## Project Overview
The goal is to build a binary classification model that predicts whether a customer will churn (leave the company) or not. The project involves a complete data science pipeline:

1.  **Data Cleaning and Preprocessing:** Handling missing values and transforming data into a usable format.
2.  **Exploratory Data Analysis (EDA):** Visualizing data to uncover patterns and relationships between features and customer churn.
3.  **Feature Engineering:** Converting categorical features into a numerical format suitable for a neural network.
4.  **Model Building:** Creating a sequential neural network using TensorFlow and Keras.
5.  **Model Evaluation:** Assessing the model's performance on unseen test data using metrics like accuracy, precision, recall, and a confusion matrix.

---

## Dataset
The dataset used is the **Telco Customer Churn** dataset, which contains information about 7,043 customers and 21 attributes, including:
* **Demographic Info:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, etc.
* **Account Information:** `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Target Variable:** `Churn` (Yes/No)

---

## Workflow
The project follows these key steps:

1.  **Load Data:** The CSV file is loaded into a pandas DataFrame.
2.  **Data Cleaning:**
    * The `customerID` column is dropped as it's not a useful feature.
    * The `TotalCharges` column is converted from `object` to a numeric type. Empty string values are identified and the corresponding rows (11 in this case) are removed.
3.  **Exploratory Data Analysis (EDA):**
    * Histograms are created to visualize the relationship between `tenure` and `Churn`, and `MonthlyCharges` and `Churn`. This revealed that customers with shorter tenures and higher monthly charges are more likely to churn.
4.  **Feature Engineering & Preprocessing:**
    * Categorical columns with 'Yes' and 'No' values are converted to `1` and `0`.
    * Values like 'No internet service' or 'No phone service' are standardized to 'No' to simplify the feature space.
    * Categorical columns with more than two unique values (`InternetService`, `Contract`, `PaymentMethod`) are one-hot encoded using `pd.get_dummies`.
    * Numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled to a range between 0 and 1 using `MinMaxScaler` to help the model converge faster.
5.  **Model Building & Training:**
    * The data is split into training (80%) and testing (20%) sets.
    * A neural network is built using TensorFlow/Keras.
    * The model is compiled with the Adam optimizer, `binary_crossentropy` loss function (ideal for binary classification), and is trained for 100 epochs.

---

## Model Architecture
A simple yet effective sequential neural network was designed with the following layers:

* **Input Layer:** A dense layer with 26 neurons (matching the number of input features) and a `ReLU` activation function.
* **Hidden Layer:** A dense layer with 15 neurons and a `ReLU` activation function.
* **Output Layer:** A dense layer with 1 neuron and a `Sigmoid` activation function, which outputs a probability between 0 and 1.

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```


**Confusion Matrix**

The confusion matrix shows the number of correct and incorrect predictions.

1. True Negatives (Top-Left): 862 customers correctly predicted as not churning.

2. False Positives (Top-Right): 137 customers incorrectly predicted as churning.

3. False Negatives (Bottom-Left): 179 customers incorrectly predicted as not churning.

4. True Positives (Bottom-Right): 229 customers correctly predicted as churning.


**Requirements**
To run this project, you need Python 3 and the following libraries:

1. pandas
2. numpy
3. matplotlib
4. scikit-learn
5. tensorflow
6. seaborn


You can install them using pip:

```python
pip install pandas numpy matplotlib scikit-learn tensorflow seaborn

```

How to Run
1. Clone the repository or download the source code.
3. Place the dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) in the same directory as your script or Jupyter Notebook.
4. Install the required libraries as listed above.
5. Execute the Python script or run the cells in the Jupyter Notebook.

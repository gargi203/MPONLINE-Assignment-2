# Customer Churn Prediction using Logistic Regression

**Author:** Gargi Sharma

**Registration Number:** 23BCE10655  

**Application Number:** IN26011777

**Batch Number:** 1A

**Email ID:** gargi.23bce10655@vitbhopal.ac.in

## Objective
The objective of this assignment is to develop a Logistic Regression model to predict whether a telecommunications customer is likely to leave (churn) based on their demographic information and service usage.

## Dataset Link
The dataset used for this project is the **Telco Customer Churn Dataset**.
Following submission guidelines, the dataset is not hosted in this repository. You can download it directly from Kaggle:
[Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> Note: The raw dataset file is intentionally excluded from this repository. Only the dataset link is provided here.

## Libraries Used
* **Pandas:** Data manipulation, cleaning, and one-hot encoding.
* **NumPy:** Numerical operations.
* **Matplotlib & Seaborn:** Data visualization and generating the confusion matrix heatmap.
* **Scikit-learn:** Splitting the dataset, building the Logistic Regression model, and evaluating performance (Accuracy, Precision, Recall, F1-Score).

## Methodology
1. **Data Understanding:** Loaded the dataset and identified numerical features (e.g., `tenure`, `MonthlyCharges`), categorical features (e.g., `Contract`, `PaymentMethod`), and the target variable (`Churn`).
2. **Data Preprocessing:** Handled missing values in the `TotalCharges` column by converting blank spaces to NaN and dropping the affected rows. Dropped the irrelevant `customerID` column. Applied One-Hot Encoding to categorical variables and mapped the target variable to binary (1 and 0). Split the data into 80% training and 20% testing sets.
3. **Model Development:** Instantiated and trained a Logistic Regression model on the training data.
4. **Model Evaluation:** Evaluated the model against the testing set, computing Accuracy, Precision, Recall, and F1-Score. Generated a confusion matrix to visualize True Positives, False Positives, True Negatives, and False Negatives.

## Results
* **Accuracy Score:** ~0.81 (81%)
* **Precision:** ~0.67
* **Recall:** ~0.56
* **F1-Score:** ~0.61

## Conclusion (Task 5)
The Logistic Regression model produced solid overall accuracy but lower recall, which means it misses a meaningful share of actual churn cases. Month-to-month contracts, higher monthly charges, and missing bundled services are the strongest churn drivers identified in this dataset. Logistic Regression is limited here because it assumes linear separability in log-odds space and cannot easily model more complex nonlinear relationships between customer features and churn risk.

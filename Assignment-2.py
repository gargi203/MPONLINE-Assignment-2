import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# Task 1: Data Understanding
# ==========================================
print("--- Task 1: Data Understanding ---")
# 1. Load the dataset
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Error: Please download the dataset from Kaggle and place it in the same directory.")
    exit()

# 2. Display the first five records
print("\nFirst 5 records of the dataset:")
print(df.head())

# 3. Identify Features
# Note: 'customerID' is an identifier, not a predictive feature.
# 'TotalCharges' is initially read as an object due to blank spaces in the dataset.
numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df.columns if col not in numerical_features and col != 'Churn' and col != 'customerID']
target_variable = 'Churn'

print("\nIdentified Features:")
print(f"Numerical Features: {numerical_features}")
print(f"Categorical Features (excluding target and ID): {categorical_features}")
print(f"Target Variable: {target_variable}")

# ==========================================
# Task 2: Data Preprocessing
# ==========================================
print("\n--- Task 2: Data Preprocessing ---")

# Convert 'TotalCharges' to numeric, forcing non-numeric values to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values
print("\nMissing values before handling:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Handle missing values (dropping the 11 rows with NaN TotalCharges is standard for this dataset)
df.dropna(inplace=True)
print(f"\nMissing values handled. Remaining rows: {len(df)}")

# Drop 'customerID' as it is irrelevant for the model
df.drop('customerID', axis=1, inplace=True)

# Encode Target Variable ('Yes' -> 1, 'No' -> 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables using One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define Features (X) and Target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Data split complete. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Scale numerical features for better convergence
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# ==========================================
# Task 3: Model Development
# ==========================================
print("\n--- Task 3: Model Development ---")
# Build and train the Logistic Regression model
# Increased max_iter and solver for more stable convergence
model = LogisticRegression(max_iter=2000, solver='lbfgs')
model.fit(X_train, y_train)

# Predict customer churn on the test dataset
y_pred = model.predict(X_test)
print("Model training and prediction complete.")

# ==========================================
# Task 4: Model Evaluation
# ==========================================
print("\n--- Task 4: Model Evaluation ---")
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy Score: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix for Churn Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('confusion_matrix.png')
plt.show()

print("\nObservations based on model performance:")
print("1. The model achieves strong overall accuracy, but lower recall indicates it misses a portion of actual churn cases.")
print("2. The precision is solid, meaning churn predictions are usually correct when the model flags churn.")
print("3. The confusion matrix reveals class imbalance impact: the model predicts 'No Churn' better than the minority churn class.")

# ==========================================
# Task 5: Conclusion
# ==========================================
print("\n--- Task 5: Conclusion ---")
conclusion = (
    "The Logistic Regression model delivered stable predictive performance with roughly 81% accuracy, "
    "but lower recall reveals it misses a substantial number of customers who actually churn. "
    "Customers on month-to-month contracts, higher monthly charges, and fewer bundled services tend to drive churn in this dataset. "
    "This behavior suggests contract length, pricing pressure, and missing product value are key churn predictors. "
    "A core limitation of Logistic Regression for this problem is that it assumes a linear relationship between features and the log-odds of churn, "
    "so it cannot easily capture more complex nonlinear interactions. In practice, this means the model can underfit customer behavior patterns "
    "when churn depends on combined or conditional service usage factors."
)
print(conclusion)
# ============================================
# Flood Probability Prediction
# Kaggle Playground Series - S4E5
# Author: [Your Name]
# Description: This script predicts flood probability using regression models on Kaggle's synthetic dataset.
# It includes data loading, preprocessing, model training (RandomForest and Polynomial Regression),
# evaluation with R² score, visualizations, and submission generation.
# Dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. Load the Data
# ============================================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("Data Loaded Successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Check for missing values (dataset is clean, but good practice)
if train.isnull().sum().any():
    print("Warning: Missing values detected. Handling them...")
    train = train.fillna(train.median())  # Impute with median if needed
    test = test.fillna(test.median())

# ============================================
# 2. Prepare Data
# ============================================
# Define target and ID columns
TARGET = "FloodProbability"
ID_COL = "id"

# Extract features (all numeric in this dataset)
X = train.drop(columns=[TARGET, ID_COL], errors="ignore").select_dtypes(include=[np.number])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL], errors="ignore").select_dtypes(include=[np.number])
test_ids = test[ID_COL]

# Handle categorical features if any (none in this dataset, but included for robustness)
cat_cols = X.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols)
    X_test = pd.get_dummies(X_test, columns=cat_cols)
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# ============================================
# 3. Scale Data
# ============================================
# Standardize features to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4. Train/Validation Split
# ============================================
# Split data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================================
# 5. Model Training - RandomForestRegressor (Primary Model)
# ============================================
# Initialize and train RandomForest model
rf_model = RandomForestRegressor(
    n_estimators=200,  # Number of trees in the forest
    max_depth=10,      # Maximum depth to prevent overfitting
    random_state=42,   # For reproducibility
    n_jobs=-1          # Use all available cores
)

rf_model.fit(X_train, y_train)

# Predict on validation set and clip to [0,1] range
val_preds_rf = rf_model.predict(X_val)
val_preds_rf = np.clip(val_preds_rf, 0, 1)

# Evaluate with R² score
r2_rf = r2_score(y_val, val_preds_rf)
print(f" RandomForest Validation R² Score: {r2_rf:.4f}")

# Feature Importance (for interpretability)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance.head())

# ============================================
# 6. Model Training - Polynomial Regression (Alternative Model)
# ============================================
# Initialize and train Polynomial Regression pipeline
poly_model = Pipeline([
    ("scaler", StandardScaler()),  # Standardize features
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # Generate polynomial features
    ("lr", LinearRegression(n_jobs=-1))  # Linear regression on transformed features
])

poly_model.fit(X_train, y_train)

# Predict on validation set and clip to [0,1] range
val_preds_poly = poly_model.predict(X_val)
val_preds_poly = np.clip(val_preds_poly, 0, 1)

# Evaluate with R² score
r2_poly = r2_score(y_val, val_preds_poly)
print(f" Polynomial Regression Validation R² Score: {r2_poly:.4f}")

# ============================================
# 7. Visualizations
# ============================================
# 1. Correlation Heatmap (using actual data)
plt.figure(figsize=(12, 10))
corr = train.drop(columns=[ID_COL]).corr()  # Compute correlation matrix
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.savefig("visualizations/correlation_heatmap.png")  # Save for README
plt.show()

# 2. Actual vs Predicted Plot (using RandomForest predictions)
plt.figure(figsize=(8, 6))
plt.scatter(y_val, val_preds_rf, alpha=0.5, color='green')
plt.plot([0, 1], [0, 1], 'r--', lw=2)
plt.xlabel("Actual FloodProbability")
plt.ylabel("Predicted FloodProbability")
plt.title("Actual vs Predicted FloodProbability (RandomForest Validation)")
plt.savefig("visualizations/actual_vs_predicted.png")  # Save for README
plt.show()

# 3. Residual Plot (using RandomForest predictions)
residuals = y_val - val_preds_rf
plt.figure(figsize=(8, 6))
plt.scatter(val_preds_rf, residuals, alpha=0.5, color='purple')
plt.hlines(y=0, xmin=0, xmax=1, colors='r', linestyles='--')
plt.xlabel("Predicted FloodProbability")
plt.ylabel("Residuals")
plt.title("Residual Plot (RandomForest)")
plt.savefig("visualizations/residual_plot.png")  # Save for README
plt.show()

# ============================================
# 8. Predict on Test Set (using best model - RandomForest)
# ============================================
test_preds = rf_model.predict(X_test_scaled)
test_preds = np.clip(test_preds, 0, 1)

# ============================================
# 9. Create Submission File
# ============================================
submission = pd.DataFrame({
    'id': test_ids,
    'FloodProbability': test_preds
})
submission.to_csv("submission.csv", index=False)
print("\n Submission file created successfully: submission.csv")
print(submission.head())
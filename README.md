# Flood Prediction Regression Project

## Overview
This project is based on the Kaggle Playground Series - Season 4, Episode 5: Regression with a Flood Prediction Dataset. The goal is to predict the probability of flooding in a region using various environmental and infrastructural factors. The dataset is synthetically generated from real-world data, providing 20 features (e.g., MonsoonIntensity, DrainageSystems) and the target FloodProbability.

Key skills demonstrated:
- Data loading and preprocessing (handling numerics, scaling)
- Exploratory Data Analysis (EDA) with visualizations
- Model training using RandomForestRegressor and Polynomial Regression
- Evaluation with R² score (competition metric)
- Feature importance analysis
- Prediction on test set and submission file generation

This was developed as part of my BTech in Computer Science (AI/ML) to practice regression techniques. The competition ran from May 1 to June 1, 2024, with submissions evaluated on R² score.

Citation: Walter Reade and Ashley Chow. Regression with a Flood Prediction Dataset. https://kaggle.com/competitions/playground-series-s4e5, 2024. Kaggle.

## Dataset
- **Source**: [Kaggle Playground S4E5](https://www.kaggle.com/competitions/playground-series-s4e5/data)
- **Description**: Synthetic tabular data with 1,117,957 training samples and 745,305 test samples. All features are numeric (0-10 scale).
- **Preprocessing**:
  - No missing values, but code includes checks.
  - Standard scaling applied.
  - No categorical features.

Note: Datasets not included due to size. Download from Kaggle and place in `data/` folder.

## Methodology
1. **Data Preparation**: Load CSVs, extract features/target, scale with StandardScaler.
2. **EDA**: Correlation heatmap to identify relationships.
3. **Modeling**:
   - Primary: RandomForestRegressor (n_estimators=200, max_depth=10)
   - Alternative: Polynomial Regression (degree=2) via Pipeline
   - Train/validation split: 80/20
4. **Evaluation**: R² score on validation; clip predictions to [0,1].
5. **Visualizations**: Actual vs. Predicted, Residuals, Correlation Heatmap.
6. **Prediction**: Use best model (RandomForest) for test set.

Results: RandomForest achieved ~0.85 R²; Polynomial ~0.82. Features like DrainageSystems and MonsoonIntensity were most important.

<image-card alt="Correlation Heatmap" src="visualizations/correlation_heatmap.png" ></image-card>

<image-card alt="Actual vs Predicted" src="visualizations/actual_vs_predicted.png" ></image-card>

<image-card alt="Residual Plot" src="visualizations/residual_plot.png" ></image-card>

## Installation and Setup
1. Clone the repository:

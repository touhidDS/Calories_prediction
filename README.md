# Calorie Burn Prediction

## Overview
This repository contains a Jupyter Notebook for predicting calories burned during exercise using a dataset from a Kaggle competition. The notebook includes data exploration, preprocessing, feature engineering, and training ensemble models (XGBoost + LightGBM) with KFold cross-validation, achieving low RMSLE via averaging predictions.

## Dataset
- **Training Data**: `/kaggle/input/playground-series-s5e6/train.csv` (750,000 entries)
- **Test Data**: `/kaggle/input/playground-series-s5e6/test.csv` (250,000 entries)
- **Features**:
  - `Sex` (categorical: male/female)
  - `Age` (numerical)
  - `Height` (numerical, cm)
  - `Weight` (numerical, kg)
  - `Duration` (numerical, minutes)
  - `Heart_Rate` (numerical)
  - `Body_Temp` (numerical, Celsius)
- **Target**: `Calories` (numerical, regression task)
- **Missing Values**: Handled by filling with column means.
- **Additional Feature**: `BMI` = Weight / (Height/100)^2

## Approach
1. **Data Loading & Exploration**:
   - Load train/test with Pandas; drop 'id'.
   - View head, info, describe, null percentages.
   - Visualizations: Boxplots for outliers, pairplots, correlation heatmap.

2. **Preprocessing**:
   - Label encode 'Sex' (male=0, female=1).
   - Fill nulls with means for numerical columns.

3. **Feature Engineering**:
   - Calculate and add 'BMI' column.

4. **Modeling**:
   - Use KFold (5 splits) for cross-validation.
   - Models: XGBRegressor (n_estimators=1000, learning_rate=0.03, etc.) and LGBMRegressor (similar params).
   - Train on folds, predict on validation/test.
   - Ensemble: Average predictions from XGBoost and LightGBM.

5. **Evaluation**:
   - Metric: Root Mean Squared Logarithmic Error (RMSLE) on validation sets.

6. **Submission**:
   - Generate `submission.csv` with averaged predictions.

## Requirements
- Python 3.11+
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (LabelEncoder, KFold, metrics), `xgboost`, `lightgbm`

Install via:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

## Results
- Ensemble approach minimizes RMSLE.
- Sample submission tail (last 10 predictions shown in notebook).

## Notes
- Designed for Kaggle environment 
- For local runs, update file paths.
- Potential improvements: Hyperparameter tuning, more features, other ensembles.


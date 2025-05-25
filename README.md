# Life Expectancy Prediction

A machine learning regression project that predicts life expectancy using various algorithms and WHO global health data.

## Project Description

This project explores multiple regression models to predict life expectancy based on health, economic, and demographic factors. The goal is to find the most accurate model through systematic comparison and hyperparameter optimization.

**Dataset**: [WHO Life Expectancy Data](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

## Methodology

1. **Feature Selection**: Correlation analysis to identify relevant predictors
2. **Model Development**: Progressive approach from simple to complex models
3. **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV for optimization
4. **Model Evaluation**: Mean Absolute Error (MAE) as primary metric

## Models Implemented

- Linear Regression (baseline)
- Ridge & Lasso Regression (regularization)
- K-Nearest Neighbors (non-linear approach)
- XGBoost (gradient boosting)

## Results

| Model | MAE (Years) | Optimization |
|-------|-------------|--------------|
| Linear Regression | 3.5 | - |
| Ridge/Lasso | 3.5 | GridSearchCV |
| KNN | 1.49 | GridSearchCV |
| XGBoost | 1.29 | - |
| **XGBoost** | **1.25** | **RandomizedSearchCV** |

**Final model achieves an average prediction error of only 1 year and 3 months.**

## Project Structure

```
LifeExpectancy/
├── get_data.py              # Data acquisition and preprocessing
├── life_regression_model.py # Linear regression implementation
├── regression_models.py     # Ridge and Lasso regression
├── knn_model.py            # K-Nearest Neighbors model
├── xgb_model.py            # XGBoost with hyperparameter tuning
├── main.py                 # Main execution pipeline
├── final_model.pkl         # Saved best model
└── requirements.txt        # Project dependencies
```

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py

# Load and use trained model
import joblib
model = joblib.load('final_model.pkl')
predictions = model.predict(X_test)
```

## Key Technologies

- **scikit-learn**: Core ML algorithms
- **XGBoost**: Gradient boosting
- **pandas & numpy**: Data manipulation
- **matplotlib**: Visualization

## Conclusions

The project demonstrates that ensemble methods (XGBoost) significantly outperform traditional linear models for life expectancy prediction. The final model's error of 1.25 years represents a 64% improvement over the baseline.

## Author

Marina Kurland
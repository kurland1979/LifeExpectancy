from get_data import get_file, clean_data
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
df = get_file()
df = clean_data(df)

def feature_testing(df):
    feature_corr = df.corrwith(df['life_expectancy'])
    return feature_corr

def split_train_test(df,columns=None):
    if columns is  None:
        columns = ['status','adult_mortality','alcohol','percentage_expenditure',
                'bmi','polio','hiv_aids','thinness__1-19_years','thinness_5-9_years']
    
    X = df[columns]    
    y = df['life_expectancy']
        
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)
    
    return X_train,X_test,y_train,y_test

def train_model(X_train,X_test,y_train,y_test):
    model_linear = Pipeline([
            ('standardscaler', StandardScaler()),
            ('linearregression', LinearRegression())
    ])
   
    model_linear.fit(X_train,y_train)
    
    y_pred = model_linear.predict(X_test)
    
    linear_mse = mean_squared_error(y_test,y_pred)
    linear_mae = mean_absolute_error(y_test,y_pred)
    linear_r2 = r2_score(y_test,y_pred)
    
    result = f'MSE: {linear_mse:.4f} \nMAE: {linear_mae:.4f} \nR2_Score: {linear_r2:.4f}'
    
    return model_linear, result,  linear_mae, linear_mse, linear_r2,y_train,y_test

def baseline_evaluation(y_train, y_test):
    baseline_pred = np.mean(y_train)
    baseline_prediction = np.full(len(y_test), baseline_pred)
    baseline_mae = mean_absolute_error(y_test, baseline_prediction)
    baseline_mse = mean_squared_error(y_test, baseline_prediction)
    baseline_r2 = r2_score(y_test, baseline_prediction)
    
    return baseline_mae, baseline_mse, baseline_r2

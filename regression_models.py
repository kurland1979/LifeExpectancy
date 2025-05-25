from get_data import get_file, clean_data
from life_regression_model import split_train_test
from sklearn.linear_model import  Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = get_file()
df = clean_data(df)

def train_ridge_model(df):
    X_train,X_test,y_train,y_test = split_train_test(df)
    
    model_ridge = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
                ])
    
    model_ridge.fit(X_train,y_train)
    
    y_pred = model_ridge.predict(X_test)
    
    ridge_mse = mean_squared_error(y_test,y_pred)
    ridge_mae = mean_absolute_error(y_test,y_pred)
    ridge_r2 = r2_score(y_test,y_pred)
    
    result_ridge = f'Ridge_Model :\n MSE: {ridge_mse:.4f} \nMAE: {ridge_mae:.4f} \nR2_Score: {ridge_r2:.4f}'
    
    
    
    return model_ridge, result_ridge,ridge_mae, ridge_mse, ridge_r2

def train_grid_ridge(model_ridge,X_train,y_train,X_test,y_test):
    params_grid = { "ridge__alpha": [0.01, 0.1, 1, 10, 100] }
    
    grid_ridge = GridSearchCV(
                estimator=model_ridge,
                param_grid=params_grid,
                cv=5
    )
    grid_ridge.fit(X_train,y_train)
    
    y_pred_grid = grid_ridge.predict(X_test)
    
    grid_ridge_mse = mean_squared_error(y_test,y_pred_grid)
    grid_ridge_mae = mean_absolute_error(y_test,y_pred_grid)
    grid_ridge_r2 = r2_score(y_test,y_pred_grid)  

    result_ridge_grid = f'Grid_Ridge_Model :\n MSE: {grid_ridge_mse:.4f} \nMAE: {grid_ridge_mae:.4f} \nR2_Score: {grid_ridge_r2:.4f}'

    return result_ridge_grid, grid_ridge,grid_ridge_mae,grid_ridge_mse,grid_ridge_r2

def train_lasso_model(df):
    X_train,X_test,y_train,y_test = split_train_test(df)

    model_lasso = Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=1.0))
                ])
    model_lasso.fit(X_train,y_train)
    
    y_pred = model_lasso.predict(X_test)
    lasso_mse = mean_squared_error(y_test,y_pred)
    
    lasso_mae = mean_absolute_error(y_test,y_pred) 
    
    lasso_r2 = r2_score(y_test,y_pred)
    
    result_lasso = f' Lasso_Model:\n MSE: {lasso_mse:.4f} \nMAE: {lasso_mae:.4f} \nR2_Score: {lasso_r2:.4f}'
    
    return model_lasso, result_lasso,lasso_mae, lasso_mse, lasso_r2


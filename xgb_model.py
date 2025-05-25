from get_data import get_file, clean_data
from life_regression_model import split_train_test
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import  RandomizedSearchCV


df = get_file()
df = clean_data(df)

def train_xgb_model(df):
    X_train,X_test,y_train,y_test = split_train_test(df)
    
    model_xgb = XGBRegressor()
    model_xgb.fit(X_train,y_train)
    
    y_pred = model_xgb.predict(X_test)
    
    xgb_mse = mean_squared_error(y_test,y_pred)
    xgb_mae = mean_absolute_error(y_test,y_pred)
    xgb_r2 = r2_score(y_test,y_pred)
    
    xgb_result = f'Model_XGB\nMSE: {xgb_mse:.4f}\nMAE: {xgb_mae:.4f}\nR2_Score: {xgb_r2:.4f}'
    
    return model_xgb,xgb_result,xgb_mse,xgb_mae,xgb_r2

def train_search_xgb(model_xgb,X_train,y_train,X_test,y_test):
    param_dist = {
                'n_estimators': [50,100,200,300,500],
                'learning_rate': [0.01,0.05,0.1,0.2,0.3],
                'max_depth': [2,4,6,8,10],
                'min_child_weight': [5,7,10]
    }
    
    search_xgb = RandomizedSearchCV(estimator=model_xgb,
                            param_distributions=param_dist,
                            cv=5,
                            random_state=42)
    
    search_xgb.fit(X_train,y_train)
    
    y_pred = search_xgb.predict(X_test)
    
    search_xgb_mse = mean_squared_error(y_test,y_pred)
    search_xgb_mae = mean_absolute_error(y_test,y_pred)
    search_xgb_r2 = r2_score(y_test,y_pred)
    
    result_search_xgb = f'Model_SEARCH_XGB\nMSE: {search_xgb_mse:.4f}\nMAE: {search_xgb_mae:.4f}\nR2_Score: {search_xgb_r2:.4f}' 
    
    return search_xgb, search_xgb_mse, search_xgb_mae, search_xgb_r2, result_search_xgb
    
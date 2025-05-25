from get_data import get_file, clean_data
from life_regression_model import split_train_test
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = get_file()
df = clean_data(df)

def train_knn_model(df):
    X_train,X_test,y_train,y_test = split_train_test(df)
    
    model_knn = KNeighborsRegressor()
    model_knn.fit(X_train,y_train)
    
    knn_pred = model_knn.predict(X_test)
    
    knn_mse = mean_squared_error(y_test,knn_pred)
    knn_mae = mean_absolute_error(y_test,knn_pred)
    knn_r2 = r2_score(y_test,knn_pred)
    knn_resut = f'Model_KN\nMSE: {knn_mse:.4f}\nMAE: {knn_mae:.4f}\nR2_Score: {knn_r2:.4f}'
    
    return model_knn, knn_mse,knn_mae,knn_r2,knn_resut
    
    
def train_grid_knn(model_knn,X_train,y_train,X_test,y_test):
    param_grid = {'n_neighbors': [5,9,11,15]}
    
    grid_knn = GridSearchCV(
            estimator=model_knn, 
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            cv=5
    )
    grid_knn.fit(X_train,y_train)
    
    grid_pred = grid_knn.predict(X_test)
    
    grid_mse = mean_squared_error(y_test,grid_pred)
    grid_mae = mean_absolute_error(y_test,grid_pred)
    grid_r2 = r2_score(y_test,grid_pred)
    
    grid_result = f'Model_Grid_KNN\nMSE: {grid_mse:.4f}\nMAE: {grid_mae:.4f}\nR2_Score: {grid_r2:.4f}'   

    return grid_knn, grid_mse,grid_mae,grid_r2,grid_result



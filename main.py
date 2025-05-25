from get_data import get_file,clean_data
from life_regression_model import split_train_test, train_model, baseline_evaluation
from regression_models import train_ridge_model, train_lasso_model, train_grid_ridge
from knn_model import train_knn_model,train_grid_knn
from xgb_model import train_xgb_model,train_search_xgb
import pandas as pd
import matplotlib.pyplot as plt

if __name__=='__main__':
    df = get_file()
    df = clean_data(df)
    
    X_train,X_test,y_train,y_test = split_train_test(df)
    
    model_linear, result_linear, linear_mae, linear_mse, linear_r2,y_train,y_test = train_model(X_train,X_test,y_train,y_test)
    
    print(result_linear)
    print('*'* 30)
    
    baseline_mae, baseline_mse, baseline_r2 = baseline_evaluation(y_train, y_test)
    print(f'Baseline MAE: {baseline_mae:.4f}')


    if float(linear_mae) < float(baseline_mae):
        print('model beats the baseline!')
    else:
        print('The model has not improved — check everything again.')
    
    print('*'* 30)
    
    model_ridge, result_ridge,ridge_mae, ridge_mse, ridge_r2 = train_ridge_model(df)
    
    print(result_ridge)
    
    print('*'* 30)
    
    result_ridge_grid, grid_ridge,grid_ridge_mae,grid_ridge_mse,grid_ridge_r2 = train_grid_ridge(model_ridge,X_train,y_train,X_test,y_test )
    print(result_ridge_grid)
    print('*'* 30)
    
    
    model_lasso, result_lasso,  lasso_mae, lasso_mse, lasso_r2 = train_lasso_model(df)
    
    print(result_lasso)
    
    print('*'* 60)
    
    # Run both regular KNN and GridSearch KNN
    model_knn, knn_mse,knn_mae,knn_r2,knn_resut = train_knn_model(df)
    grid_knn, grid_mse,grid_mae,grid_r2,grid_result = train_grid_knn(model_knn, X_train, y_train, X_test, y_test)
    
    if grid_mae < knn_mae:
        print("GridSearch improved the result:")
        print(grid_result)
        print(f"Grid KNN (n_neighbors={grid_knn.best_params_['n_neighbors']}), MAE: {grid_mae:.4f}")
    else:
        print("GridSearch did not improve performance. Default parameter was optimal.")
        print('-'*40)
        print(grid_result)
        print(f"KNN (n_neighbors={grid_knn.best_params_['n_neighbors']}) and GridSearch are identical, MAE: {knn_mae:.4f}")
        print('*'*30)
        
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print('-'*40)

    # compare to KNN/Grid
    if grid_mae < baseline_mae:
        print(f"Grid KNN improved MAE compared to baseline by {(baseline_mae - grid_mae):.2f}")
    else:
        print("Grid KNN did not improve compared to baseline.")

    print('*'* 30)
    model_xgb,xgb_result,xgb_mse,xgb_mae,xgb_r2 = train_xgb_model(df)
    search_xgb, search_xgb_mse, search_xgb_mae, search_xgb_r2, result_search_xgb = train_search_xgb(model_xgb,X_train, y_train, X_test, y_test)
    
    if search_xgb_mae < xgb_mae:
        print("RandomizedSearch improved the result:")
        print(result_search_xgb)
    else:
        print("RandomizedSearch did not improve performance. Default parameter was optimal.")
        print('-'*40)
        print(xgb_result)
        print('*'* 30)
    
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print('-'*40)
    
    if search_xgb_mae < baseline_mae:
        print(f"RandomizedSearch improved MAE compared to baseline by {(baseline_mae - search_xgb_mae):.2f}")
        print('-'*40)
    else:
        print("RandomizedSearch did not improve compared to baseline.")
    
    # Fill with your actual results
    
    data = [
        {'Model':'Linear','MAE': linear_mae,'MSE': linear_mse,'R2': linear_r2 },
        {'Model':'Ridge','MAE': ridge_mae,'MSE': ridge_mse,'R2': ridge_r2 },
        {'Model':'GridSearch_Ridge','MAE': grid_ridge_mae,'MSE': grid_ridge_mse,'R2': grid_ridge_r2 },
        {'Model':'Lasso','MAE': lasso_mae,'MSE': lasso_mse,'R2': lasso_r2 },
        {'Model':'KNeighbors','MAE': knn_mae,'MSE': knn_mse,'R2': knn_r2 },
        {'Model':'GridSearch_KNN','MAE': grid_mae,'MSE': grid_mse,'R2': grid_r2 },
        {'Model':'XGB','MAE': xgb_mae,'MSE': xgb_mse,'R2': xgb_r2 },
        {'Model':'Search_XGB','MAE': search_xgb_mae,'MSE': search_xgb_mse,'R2': search_xgb_r2 },
        {'Model':'Baseline','MAE': baseline_mae,'MSE': baseline_mse,'R2': baseline_r2}     
    ]
   
    data_df = pd.DataFrame(data)
   
    print(data_df)
          
    data_df_sorted = data_df.sort_values(by='MAE', ascending=True)

    font1 = {'family':'serif','color':'blue','size':20}
    font2 = {'family':'serif','color':'darkred','size':15}
    
    plt.bar(x=data_df_sorted['Model'], height=data_df_sorted['MAE'],color='r')
    plt.title("Model Comparison by MAE", fontdict=font1)
    plt.xlabel('Model',fontdict=font2)
    plt.ylabel('MAE', fontdict=font2)
    for i, v in enumerate(data_df_sorted['MAE']):
        plt.text(x=i, y=v + 0.02, s=round(v, 2), ha='center', fontdict=font2)
    plt.show()
    
    print("""
    Project Summary:

    The features were selected based on correlation analysis.
    Initially, a simple Linear Regression model was built, but its performance was not impressive.
    In parallel, two more models, Ridge and Lasso, were trained, but they did not improve the results, even after adding a GridSearch wrapper. In my opinion, this is because there was no strong relationship between the features.
    Then, a KNN model was built, which significantly improved the MAE result to 1.49. 
    Adding GridSearch to it did not change the result at all.
    Despite this decent result, I decided to try an XGB model, which further improved the prediction by 0.79 compared to KNN.
    Of course, I also applied a RandomizedSearch wrapper to XGB, which gave a small additional improvement.
    The final decision was to keep RandomizedSearchXGB, as it provided quite accurate predictions—
    overall, the average deviation was only one year and three months in life expectancy.
    """)
    
    # save final model
    import joblib
    joblib.dump(search_xgb,'final_model.pkl')
    
    

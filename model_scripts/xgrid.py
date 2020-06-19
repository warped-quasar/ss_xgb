from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score
import xgboost as xgb
import numpy as np
from utils.master_datain import get_df_Xy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
#
# # will open features importance plots in new window
matplotlib.use('TkAgg')


df, (X, y) = get_df_Xy(dataset='master_2020_renamed', drop_unimportant=False)


# gbm_param_grid = {
#     # Parameters that we are going to tune.
#     'max_depth': [6],
#     'min_child_weight': [1],
#     'eta': [0.0244],
#     'subsample': [1],
#     'colsample_bytree': [1],
#     'n_estimators': [100]
# }

# 04/22/2020 dropping regularization params and creating wider grid for other params

# gbm_param_grid = {'learning_rate': [0.001, 0.05, 0.1, 0.2, ],
#                   'max_depth': [4, 5, 6, 8, 10, 12],
#                   'min_child_weight': [1, 3, 4, 5, 6, 7, 8, 9],
#                   'gamma': [0.0, 0.1, 0.2],
#                   'colsample_bytree': [0.4, 0.5, 0.7],
#                   'n_estimators': [10, 25, 50, 100, 250]}

#
# # TRAINING MAX_DEPTH AND MIN_CHILD_WEIGHT
# gbm_param_grid = {
#     # Parameters that we are going to tune.
#     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#     'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#     'eta': [0.02529],
#     # 'subsample': [1],
#     # 'colsample_bytree': [1],
#     'n_estimators': [100]
# }

# BEST 'max_depth': 2, 'min_child_weight': 15

# # RETRAINING N_ESTIMATORS & ETA
# gbm_param_grid = {
#  # Parameters that we are going to tune.
#  'max_depth': [2],
#  'min_child_weight': [15],
#  'eta': [0.01, 0.025, 0.05],
#  # 'subsample': [1],
#  # 'colsample_bytree': [1],
#  'n_estimators': [10, 25, 50, 100, 250, 500]
# }

# # TESTING SUBSAMPLE AND COLSAMPLE_BYTREE
# gbm_param_grid = {
#  # Parameters that we are going to tune.
#  'max_depth': [5],
#  'min_child_weight': [9],
#  'eta': [0.0225],
#  'subsample': [0.7, 0.8, 0.9, 1],
#  'colsample_bytree': [0.7, 0.8, 0.9, 1],
#  'n_estimators': [100]
# }
# BEST 'colsample_bytree': 1, 'subsample': 0.7


# # for tuning parameters
# gbm_param_grid = {
#     'colsample_bytree': [0.7, 0.8, 0.9, 1],
#     'gamma': [0, 0.1],
#     'min_child_weight': [8, 9, 10, 11, 12],
#     'learning_rate': [0.0225],
#     'max_depth': [3, 4, 5, 6, 7, 8, 9],
#     'n_estimators': [100],
#     'reg_alpha': [0, 1e-5, 0.75],
#     'reg_lambda': [0, 1e-5, 0.45],
#     'subsample': [1]
# }

"""GET BEST NESTIMATORS AND ETA"""
gbm_param_grid = {
    # Parameters that we are going to tune.
    'eta': [0.002463],
    'n_estimators': [100]
}

xgb_estimator = xgb.XGBRegressor()

gsearch = GridSearchCV(estimator=xgb_estimator,
                       param_grid=gbm_param_grid,
                       n_jobs=-1,
                       verbose=1,
                       cv=5,
                       scoring='neg_mean_absolute_error',
                       )

gsearch.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", gsearch.best_params_)
print("Lowest MAE found: ", np.abs(gsearch.best_score_))

# preds = gsearch.best_estimator_.predict(X)
# explained_variance = explained_variance_score(y, preds)
print(f"Explained Variance: {gsearch.best_score_}")

results_df = pd.DataFrame(gsearch.cv_results_)
# results_df.drop(
#     columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params', 'split0_test_score',
#              'split1_test_score'],
#                 inplace=True)

# results_df.columns = [
#     'param_colsample_bytree',
#     'param_eta',
#     'param_gamma',
#     'param_max_depth',
#     'param_min_child_weight',
#     'param_n_estimators',
#     'param_subsample',
#     'mean_test_score',
#     'std_test_score',
#     'rank_test_score']

results_df.sort_values(ascending=True, by=['rank_test_score'], axis=0, inplace=True)
results_df.reset_index(inplace=True, drop=True)

results_df = results_df.astype(float)

# results_df.to_csv('MAJOR_GRID_SEARCH.csv')

# results_df[['param_max_depth', 'param_min_child_weight']].plot()
# plt.show()
#
# results_df.to_csv('GridSearch_keep_unimportant.csv')

# overnight grid best best params
# {'colsample_bytree': 0.6, 'eta': 0.0244, 'gamma': 0, 'max_depth': 8, 'min_child_weight': 7, 'n_estimators': 100, 'subsample': 0.6}
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
from utils.master_datain import get_df_Xy
import pickle
import os
import uuid

import sqlite3
import pandas as pd

df, (X, y) = get_df_Xy(dataset='master_2020_renamed')


# run_id = uuid.uuid4().hex[:10]
# print(f'Starting proc using RUNID: {run_id}')
# run_output_dir = os.path.join(r'C:\Users\nick\PycharmProjects\ss_xgb\models\grid_model_outputs', run_id)
# os.mkdir(run_output_dir)


# gbm_param_grid = {
#     # Parameters that we are going to tune.
#     'max_depth': 6,
#     'min_child_weight': 1,
#     'eta': 0.0225,
#     'subsample': 1,
#     'colsample_bytree': 1,
# }
# 04/22/2020 dropping regularization params and creating wider grid for other params

# gbm_param_grid = {'learning_rate': [0.001, 0.05, 0.1, 0.2, ],
#                   'max_depth': [4, 5, 6, 8, 10, 12],
#                   'min_child_weight': [1, 3, 4, 5, 6, 7, 8, 9],
#                   'gamma': [0.0, 0.1, 0.2],
#                   'colsample_bytree': [0.4, 0.5, 0.7],
#                   'n_estimators': [10, 25, 50, 100, 250]}

# TESTING MAX_DEPTH AND MIN_CHILD_WEIGHT
gbm_param_grid = {
 # Parameters that we are going to tune.
 'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12],
 'min_child_weight': [1, 3, 4, 5, 6, 7, 8, 9],
 'eta': [0.025],
 'subsample': [1],
 'colsample_bytree': [1],
 'n_estimators': [100]
}

# BEST 'max_depth': 5, 'min_child_weight': 9

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

# # for tuning parameters
# gbm_param_grid = {
#     'colsample_bytree': [0.7, 0.8, 0.9, 1],
#     'gamma': [0, 0.1],
#     'min_child_weight': [8, 9, 10, 11, 12],
#     'learning_rate': [0.0225],
#     'max_depth': [3, 4, 5, 6, 7, 8, 9],
#     'n_estimators': [100],
#     'reg_alpha': [1e-5, 0.75],
#     'reg_lambda': [1e-5, 0.45],
#     'subsample': [1]
# }

xgb_estimator = xgb.XGBRegressor()

gsearch = GridSearchCV(estimator=xgb_estimator,
                       param_grid=gbm_param_grid,
                       n_jobs=-1,
                       verbose=1,
                       cv=4,
                       scoring='neg_mean_squared_error')

gsearch.fit(X, y)

preds = gsearch.predict(X)

df['preds'] = preds

# Print the best parameters and lowest RMSE
print("Best parameters found: ", gsearch.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(gsearch.best_score_)))

results_df = pd.DataFrame(gsearch.cv_results_)
results_df.sort_values(ascending=True, by=['rank_test_score'], axis=0, inplace=True)
results_df.reset_index(inplace=True)
# results_df.to_csv(f'C:\\Users\\nick\\PycharmProjects\\ss_xgb\\models\\grid_model_outputs\\{run_id}\\dropping_business_{run_id}.csv', index=False)

df['abs_error'] = abs(df['act'] - df['preds'])

df['abs_error_f'] = df['abs_error'].apply(lambda x : "$" + "{:,}".format(round(x)))
df['act_f'] = df['act'].apply(lambda x : "$" + "{:,}".format(round(x)))
df['preds_f'] = df['preds'].apply(lambda x : "$" + "{:,}".format(round(x)))

df.to_csv(f'C:\\Users\\nick\\PycharmProjects\\ss_xgb\\models\\grid_model_outputs\\{run_id}\\dropping_business_{run_id}_with_preds.csv',
          index=False)


# last_run = {'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 50}
# # save best_estimator
# pickle.dump(gsearch.best_estimator_,
#             open(f'C:\\Users\\nick\\PycharmProjects\\ss_xgb\\models\\grid_model_outputs\\{run_id}_model.pickle.dat'),
#                  "wb")
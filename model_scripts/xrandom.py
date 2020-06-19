import pickle
import time
import uuid
import os

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import explained_variance_score
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from utils.master_datain import get_df_Xy

# will open features importance plots in new window
matplotlib.use('TkAgg')

df, (X, y) = get_df_Xy(dataset='master_2020_renamed',
                       drop_unimportant=False,
                       drop_arcs=True)

# split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# parameters = {'n_estimators': [5, 10, 25, 50],
#               "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
#               "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
#               "min_child_weight": [1, 2, 3, 4, 5, 6, 7],
#               "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
#               "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#               "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#               'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}


# parameters = {'n_estimators': [4, 5, 10, 20, 50, 100],
#               "learning_rate": [0.1, 0.2],
#               "max_depth": [3, 4, 6, 8, 10],
#               "min_child_weight": [4, 5, 6, 7],
#               "gamma": [0.0, 0.1, 0.4],
#               "colsample_bytree": [0.4, 0.7, 0.8],
#               "subsample": [0.5, 0.6],
#               'reg_lambda': [1, 5, 10]}

# """zero to onehundred"""
# all_to_onehundred = [i for i in range(2, 100, 1)]
# gbm_param_grid = {
#     # Parameters that we are going to tune.
#     # 'max_depth': [5],
#     # 'min_child_weight': [14],
#     # 'eta': stats.uniform(0.001, 0.7),
#     # 'eta': stats.uniform(0.025485910912153215, 0.013408039019882773),
#     # 'subsample': [1],
#     # 'colsample_bytree': [1],
#     'n_estimators': all_to_onehundred
# }


# """THIS ONE HAS DISTRIBUTIONS IN IT USE THIS!!!!"""
gbm_param_grid = {
    # Parameters that we are going to tune.
    # 'max_depth': [5],
    # 'min_child_weight': [14],
    # 'eta': stats.uniform(0.001, 0.7),
    'eta': stats.uniform(0.25,  0.025),
    # 'subsample': [1],
    # 'colsample_bytree': [1],
    'n_estimators': [100]
}

# # TRAINING MAX_DEPTH AND MIN_CHILD_WEIGHT
# gbm_param_grid = {
#     # Parameters that we are going to tune.
#     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#     'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#     'eta': [0.056058],
#     # 'subsample': [1],
#     # 'colsample_bytree': [1],
#     'n_estimators': [30]
# }

# gbm_param_grid = {
#     # Parameters that we are going to tune.
#     'max_depth': [5],
#     'min_child_weight': [14],
#     'eta': [0.02529],
#     'subsample': stats.uniform(0.6, 0.4),
#     'colsample_bytree': stats.uniform(0.6, 0.4),
#     'n_estimators': [100]
# }

# fit_params = {'eval_metric': 'mae',
#               'early_stopping_rounds': 10,
#               'eval_set': [(X_train, y_train), (X_test, y_test)]}

gbm = xgb.XGBRegressor()


# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator=gbm,
                                    param_distributions=gbm_param_grid,
                                    scoring='neg_mean_absolute_error',
                                    cv=5,
                                    n_jobs=-1,
                                    n_iter=75,
                                    verbose=1)

# Fit randomized_mse to the data
randomized_mse.fit(X, y)

preds = randomized_mse.predict(X)
df['preds'] = preds

preds = randomized_mse.best_estimator_.predict(X)
explained_variance = explained_variance_score(y, preds)
print(f"Explained Variance: {explained_variance}")

results_df = pd.DataFrame(randomized_mse.cv_results_)
results_df.sort_values('rank_test_score', inplace=True)
results_df.reset_index(drop=True, inplace=True)

results_df['param_eta'].plot()
plt.show()

top_10_results = results_df['param_eta'].astype(float)[:10]
bottom_range, top_range = top_10_results.min(), top_10_results.max()
print((bottom_range, top_range-bottom_range))
new_range = (bottom_range, top_range-bottom_range)

# # df.to_csv(data_w_preds_fp, index=False)
#
# # Print the best parameters and lowest RMSE
# print("Best parameters found: ", randomized_mse.best_params_)
# print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
#
#
# # print_comparisons(df_w_preds=df)
# # write_comparisons(df_w_preds=df, output_fp=comparisons_fp)
#
# results_df = pd.DataFrame(randomized_mse.cv_results_)
# results_df.sort_values(ascending=True, by=['rank_test_score'], axis=0, inplace=True)
# results_df.reset_index(inplace=True)
# results_df.to_csv(csv_output_fp, index=False)
#
# param_strings_list = list(parameters.keys())
#
#
# # save best_estimator
# pickle.dump(randomized_mse.best_estimator_,
#             open(os.path.join(run_output_dir_path,f"model_{unique_run_id}.pickle.dat"),
#                  "wb"))
#
# print(f'Total Run Time: {time.time() - start_time}')
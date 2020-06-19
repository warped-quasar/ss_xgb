from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error
import xgboost as xgb
import numpy as np
from utils.master_datain import get_df_Xy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
#
# # will open features importance plots in new window
matplotlib.use('TkAgg')


df, (X, y) = get_df_Xy(dataset='master_2020_renamed',
                       drop_unimportant=False,
                       drop_arcs=True
                       )

info_df = df.loc[:, ['ns_id', 'CASS_Address', 'CASS_City', 'CASS_State', 'Territory',
                     'terr_mean', 'Total Revenue', 'rev_index']]

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.25,
#                                                     random_state=123,
#                                                     # shuffle=True
#                                                     )

"""GET BEST NESTIMATORS AND ETA"""
gbm_param_grid = {
    # Parameters that we are going to tune.
    'eta': [0.272],
    'n_estimators': [100]
}

# TRAINING MAX_DEPTH AND MIN_CHILD_WEIGHT
# gbm_param_grid = {
#     # Parameters that we are going to tune.
#     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#     'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#     'eta': [0.023979],
#     # 'subsample': [1],
#     # 'colsample_bytree': [1],
#     'n_estimators': [100]
# }




xgb_estimator = xgb.XGBRegressor()

gsearch = GridSearchCV(estimator=xgb_estimator,
                       param_grid=gbm_param_grid,
                       n_jobs=-1,
                       verbose=1,
                       cv=10,
                       # scoring='explained_variance',
                       scoring='neg_mean_absolute_error',
                       )

gsearch.fit(X, y)

# Print the best parameters and lowest score
print("Best parameters found: ", gsearch.best_params_)
print("Lowest MAE found: ", np.abs(gsearch.best_score_))

preds = gsearch.best_estimator_.predict(X)
explained_variance = explained_variance_score(y, preds)
# 'explained_variance'
print(f"Explained Variance: {explained_variance}")



results_df = pd.DataFrame(gsearch.cv_results_)
results_df.drop(
    columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params'],
                inplace=True)

results_df.sort_values(ascending=True, by=['rank_test_score'], axis=0, inplace=True)
results_df.reset_index(inplace=True, drop=True)

results_df = results_df.astype(float)

# append predictions from the whole dataset back to the original info_df
info_df['preds'] = gsearch.best_estimator_.predict(X)
info_df['abs_error'] = abs(info_df['Total Revenue'] - info_df['preds'])

info_df['Total Revenue_f'] = info_df['Total Revenue'].apply(lambda x: "$" + "{:,}".format(round(x)))
info_df['preds_f'] = info_df['preds'].apply(lambda x: "$" + "{:,}".format(round(x)))
info_df['abs_error_f'] = info_df['abs_error'].apply(lambda x: "$" + "{:,}".format(round(x)))
info_df['predicted_rev_index'] = round((info_df['preds'] / info_df['terr_mean']) * 100, 2)
info_df['abs_index_error'] = round(abs(info_df['rev_index'] - info_df['predicted_rev_index']), 2)
info_df['index_error'] = round(info_df['predicted_rev_index'] - info_df['rev_index'], 2)


output_df = info_df.loc[:, ['ns_id', 'CASS_Address', 'CASS_State', 'Territory', 'Total Revenue_f',
                            'preds_f', 'abs_error_f', 'rev_index', 'predicted_rev_index', 'abs_index_error', 'index_error']]
output_df.sort_values(by='abs_index_error', ascending=False, inplace=True)
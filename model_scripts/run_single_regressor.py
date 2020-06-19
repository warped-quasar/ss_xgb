import re
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, explained_variance_score

import matplotlib.pyplot as plt
import matplotlib

from utils.models_utils import print_comparisons
from utils.master_datain import get_df_Xy


# will open features importance plots in new window
# matplotlib.use('TkAgg')

df, (X, y) = get_df_Xy(dataset='master_2020_renamed', drop_unimportant=False, drop_arcs=True)

info_df = df.loc[:, ['ns_id', 'CASS_Address', 'CASS_ZIP', 'CASS_City', 'CASS_State', 'Territory',
                     'terr_mean', 'Total Revenue', 'rev_index']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    # random_state=123,
                                                    shuffle=True
                                                    )
n_estimators = 10000
# params = 'default'
# params = {
#  # Parameters that we are going to tune.
#  'max_depth': 4,
#  'min_child_weight': 1,
#  'eta': 0.01,
#  'subsample': 1,
#  'colsample_bytree': 1,
#  'n_estimators':n_estimators
# }

# params = {
#         'colsample_bytree': 1,
#           'gamma': 0,
#           'eta': 0.0225,
#           'max_depth': 3,
#           'min_child_weight': 12,
#           'n_estimators': 100,
#           # 'reg_alpha': 0.75,
#           # 'reg_lambda': 0.45,
#           'subsample': 1}

# # reg:linear is deprecated, use 'reg:squarederror' for regression
# xreg = xgb.XGBRegressor(objective='reg:squarederror',
#                         colsample_bytree=params['colsample_bytree'],
#                         gamma=params['gamma'],
#                         learning_rate=params['eta'],
#                         max_depth=params['max_depth'],
#                         min_child_weight=params['min_child_weight'],
#                         n_estimators=10000,
#                         subsample=params['subsample'],
#                         # reg_alpha=params['reg_alpha'],
#                         # reg_lambda=params['reg_lambda'],
#                         seed=42,
#                         verbose=True)

xreg = xgb.XGBRegressor(
    n_estimators=10000,
    # learning_rate=0.023979,
    # max_depth=5,
    # min_child_weight=14,
)
# xreg.fit(X_train, y_train, eval_metric='mae')


eval_set = [(X_test, y_test)]
xreg.fit(X_train, y_train,
         eval_metric="mae",
         eval_set=eval_set,
         early_stopping_rounds=250,
         verbose=True)

preds = xreg.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_test, preds))
# print("RMSE: %f" % (rmse))
print(f'MAE: {mean_absolute_error(y_test, preds)}')

# mae = mean_absolute_error(y_test, preds)
explained_variance = explained_variance_score(y_test, preds)
print(f"Explained Variance: {explained_variance}")




# append predictions from the whole dataset back to the original info_df
info_df['preds'] = xreg.predict(X)
info_df['abs_error'] = abs(info_df['Total Revenue'] - info_df['preds'])

info_df['Total Revenue_f'] = info_df['Total Revenue'].apply(lambda x: "$" + "{:,}".format(round(x)))
info_df['preds_f'] = info_df['preds'].apply(lambda x: "$" + "{:,}".format(round(x)))
info_df['abs_error_f'] = info_df['abs_error'].apply(lambda x: "$" + "{:,}".format(round(x)))
info_df['predicted_rev_index'] = round((info_df['preds'] / info_df['terr_mean']) * 100, 2)
info_df['abs_index_error'] = round(abs(info_df['rev_index'] - info_df['predicted_rev_index']), 2)
info_df['index_error'] = round(info_df['predicted_rev_index'] - info_df['rev_index'], 2)


output_df = info_df.loc[:, ['ns_id', 'CASS_ZIP', 'CASS_Address', 'CASS_State', 'Territory', 'Total Revenue_f',
                            'preds_f', 'abs_error_f', 'rev_index', 'predicted_rev_index', 'abs_index_error', 'index_error']]
output_df.sort_values(by='abs_index_error', ascending=False, inplace=True)

# xgb.plot_importance(xreg, max_num_features=15, grid=True, title=f'LR = {learning_rate}')
# plt.show()

# last no business best params
# {'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 50}

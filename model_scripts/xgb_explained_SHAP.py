import xgboost as xgb
import shap
from utils.data_in import get_df_Xy
from sklearn.model_selection import train_test_split

import matplotlib
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from utils.master_datain import get_df_Xy
df, (X, y) = get_df_Xy(dataset='master_2020_renamed')

# will open features importance plots in new window
matplotlib.use('TkAgg')

# # load JS visualization code to notebook
# shap.initjs()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    # random_state=123,
                                                    shuffle=True)

# params = {'colsample_bytree': 1,
#           'gamma': 0,
#           'learning_rate': 0.0225,
#           'max_depth': 3,
#           'min_child_weight': 12,
#           'n_estimators': 100,
#           'reg_alpha': 0.75,
#           'reg_lambda': 0.45,
#           'subsample': 1}

# # reg:linear is deprecated, use 'reg:squarederror' for regression
# xreg = xgb.XGBRegressor(colsample_bytree=params['colsample_bytree'],
#                         gamma=params['gamma'],
#                         learning_rate=params['learning_rate'],
#                         max_depth=params['max_depth'],
#                         min_child_weight=params['min_child_weight'],
#                         n_estimators=params['n_estimators'],
#                         subsample=params['subsample'],
#                         reg_alpha=params['reg_alpha'],
#                         reg_lambda=params['reg_lambda'],
#                         seed=42)

eta = 0.025297
# # for baseline results use empty xreg below
xreg = xgb.XGBRegressor(n_estimators=100,
                        learning_rate=eta,
                        max_depth=8,
                        min_child_weight=7,
                        # subsample=0.9,
                        # colsample_bytree=1
                        )

xreg.fit(X, y)
# load JS visualization code to notebook
# shap.initjs()

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(xreg)
shap_values = explainer.shap_values(X)
# summarize the effects of all the features
shap.summary_plot(shap_values, X)


# df['preds'] = xreg.predict(X)
# df['abs_error'] = abs(df['Total Revenue'] - df['preds'])
#
#
# df['Total Revenue_f'] = df['Total Revenue'].apply(lambda x: "$" + "{:,}".format(round(x)))
# df['preds_f'] = df['preds'].apply(lambda x: "$" + "{:,}".format(round(x)))
# df['abs_error_f'] = df['abs_error'].apply(lambda x: "$" + "{:,}".format(round(x)))


# # for i in range(0,5):
# #     print(f'Plot {i + 1} data:  {X.iloc[i,:]}')
# #     # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

# shap.force_plot(explainer.expected_value, shap_values[15,:], X.iloc[15,:], matplotlib=True,
#                 figsize=(20,5), text_rotation=45)

# shap.dependence_plot("Apparel Stores", shap_values, X)

# shap.force_plot(explainer.expected_value,
#                 shap_values[219, :],
#                 X.iloc[219, :],
#                 matplotlib=True,
#                 figsize=(20, 5),
#                 text_rotation=45)

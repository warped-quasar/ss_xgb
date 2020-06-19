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

df, (X, y) = get_df_Xy(dataset='master_2020_renamed', drop_unimportant=False)

info_df = df.loc[:, ['ns_id', 'CASS_Address', 'CASS_City', 'CASS_State', 'Territory',
                     'terr_mean', 'Total Revenue', 'rev_index']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    # shuffle=True
                                                    )

all_eta_loops = []
for i in np.arange(1, 200, 1):
    xreg = xgb.XGBRegressor(
        n_estimators=i,
        learning_rate=0.023979,
        # max_depth=9,
        # min_child_weight=16,
        # subsample=1,
        # colsample_bytree=1,
        # reg_alpha=0.75,
        # reg_lambda=0.45,
    )
    # xreg.fit(X_train, y_train, eval_metric='mae')


    eval_set = [(X_test, y_test)]
    xreg.fit(X_train, y_train,
             eval_metric="mae",
             eval_set=eval_set,
             early_stopping_rounds=25,
             verbose=True)

    preds = xreg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

    explained_variance = explained_variance_score(y_test, preds)
    print(f"Explained Variance: {explained_variance}")
    best_round = xreg.best_ntree_limit - 1

    loop_dict = {'n_trees': i,
                 'best_round': best_round,
                 'explained_variance': explained_variance}

    all_eta_loops.append(loop_dict)
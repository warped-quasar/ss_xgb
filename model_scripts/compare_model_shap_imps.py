import xgboost as xgb
import shap
from utils.data_in import get_df_Xy
import matplotlib
from sklearn.model_selection import train_test_split

import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# will open features importance plots in new window
matplotlib.use('TkAgg')

# load JS visualization code to notebook
shap.initjs()


def get_shap_comparisons(df, X, y, params):
    df, (X, y) = get_df_Xy(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=123)

    if params != 'default':
        # reg:linear is deprecated, use 'reg:squarederror' for regression
        xreg = xgb.XGBRegressor(colsample_bytree=params['colsample_bytree'],
                                gamma=params['gamma'],
                                learning_rate=params['learning_rate'],
                                max_depth=params['max_depth'],
                                min_child_weight=params['min_child_weight'],
                                n_estimators=params['n_estimators'],
                                subsample=params['subsample'],
                                reg_alpha=params['reg_alpha'],
                                reg_lambda=params['reg_lambda'],
                                seed=42)
    else:
        # # for baseline results use empty xreg below
        xreg = xgb.XGBRegressor()

    xreg.fit(X_train, y_train)

    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(xreg)
    shap_values = explainer.shap_values(X)

    # get df of the SHAP importance ranked highest to lowest
    # def get_shap_imp_ranked(shap_values):
    # sort the features indexes by their importance in the model
    # (sum of SHAP value magnitudes over the validation dataset)
    top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
    shap_feat_imps = []
    for shap_rank, shap_index in enumerate(top_inds):
        for count, colname in enumerate(X.columns.tolist()):
            if shap_index == count:
                shape_rank = shap_rank + 1
                # print(shape_rank, shap_index, x)
                shap_feat_imps.append((shape_rank, colname, shap_index))
    shape_rank_ind_col_df = pd.DataFrame(shap_feat_imps, columns=['SHAP Rank', 'Column Name', 'Original Col Index'])
    return shape_rank_ind_col_df


    # return shape_rank_ind_col_df

    # # shap_imp_rank_df = get_shap_imp_ranked(shap_values)
    # model_name = 'Base Model'
    # shap_imp_rank_df.to_csv(f'C:\\Users\\nick\\PycharmProjects\\ss_xgb\\shap_imps\\{model_name}.csv', index=False)
    #
    # def merge_and_compare_csv_outcomes():
    #     base_model = pd.read_csv(r'C:\Users\nick\PycharmProjects\ss_xgb\shap_imps\Base Model.csv')
    #     tuned_model = pd.read_csv(r'C:\Users\nick\PycharmProjects\ss_xgb\shap_imps\Tuned Model.csv')
    #
    #     merged = pd.merge(left=base_model, right=tuned_model, on='SHAP Rank')
    #     merged = merged.astype(str)
    #     merged = merged[['SHAP Rank', 'Column Name_x', 'Column Name_y', 'Original Col Index_x', 'Original Col Index_y']]
    #     merged.columns = ['SHAP Rank', 'base model colnames', 'tuned model colnames', 'Original Col Index_x', 'Original Col Index_y']
    #     return merged

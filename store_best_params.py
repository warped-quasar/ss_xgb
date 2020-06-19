import sqlite3
import pandas as pd


def store_important_params(params, title, notes, run_id):

    conn = sqlite3.connect(r"C:\Users\nick\PycharmProjects\ss_xgb\model_2020_data\model_data_2020.sqlite")

    param_df_dict = {}
    param_df_dict['run_id'] = 'e8168eac2f'
    param_df_dict['title'] = ['Dropped Ethnicities']

    params = {'colsample_bytree': 0.4,'gamma': 0.0,'learning_rate': 0.05,'max_depth': 8, 'min_child_weight': 1,'n_estimators': 50}

    param_df_dict['params'] = [f"{params}"]

    param_df_dict['notes'] = ['Removed all ethinicities and NAICS 81']
    for k, v in params.items():
        param_df_dict[k] = [v]

    param_df_w_info = pd.DataFrame(param_df_dict)

    param_df_w_info.to_sql('important_params_with_info', conn, if_exists='replace', index=False)

run_id = 'e8168eac2f'
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score, mean_absolute_error
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')

# read in all finalized modeling data
all_locations = pd.read_csv(
    r'C:\Users\nick\PycharmProjects\ss_xgb\MASTER_DATA\MARKING ARCs - Amending Master Data.csv'
)
all_locations['CASS_ZIP'] = all_locations['CASS_ZIP'].astype(str).str.zfill(5)

def get_df_Xy(all_locations):
    non_data_columns = ['CASS_Address',
                        'ARC',
                        'Store',
                        'CASS_City',
                        'CASS_State',
                        'CASS_ZIP',
                        'CASS_Plus4',
                        'Territory',
                        'Store Present pct',
                        'terr_mean',
                        'rev_index',
                        'ns_id',
                        'Right_Territory',
                        'Username',
                        'Location ID',
                        'image_path'
                        ]

    unwanted_data_cols = ['ests_pct Other Services (except Public Administration) (81)',
                          'emps_pct Other Services (except Public Administration) (81)']

    all_locations.drop(columns=unwanted_data_cols, inplace=True)
    # info_df = all_locations.copy()

    all_locations.drop(columns=non_data_columns, inplace=True)

    arcs = all_locations.loc[all_locations['arc_flag'] == True]
    no_arcs = all_locations.loc[all_locations['arc_flag'] == False]

    all_locations.drop(columns=['arc_flag'], inplace=True)
    arcs.drop(columns=['arc_flag'], inplace=True)
    no_arcs.drop(columns=['arc_flag'], inplace=True)

    # al_return_df, (al_X, al_y) = get_df_Xy(all_locations)
    # arc_return_df, (arc_X, arc_y) = get_df_Xy(all_locations)
    # noarc_return_df, (al_X, al_y) = get_df_Xy(all_locations)

    all_locations_X, all_locations_y = all_locations.iloc[:, :-1], all_locations.iloc[:, -1]
    arcs_X, arcs_y = arcs.iloc[:, :-1], arcs.iloc[:, -1]
    noarcs_X, noarcs_y = no_arcs.iloc[:, :-1], no_arcs.iloc[:, -1]
    return [(all_locations, (all_locations_X, all_locations_y)),
            (arcs, (arcs_X, arcs_y )),
            (no_arcs, (noarcs_X, noarcs_y))]


[(all_locations, (all_locations_X, all_locations_y)),
            (arcs, (arcs_X, arcs_y )),
            (no_arcs, (noarcs_X, noarcs_y))] = get_df_Xy(all_locations)


dmatrix = xgb.DMatrix(data=noarcs_X, label=noarcs_y)
untuned_params = {"objective": "reg:squarederror"}

untuned_cv_results_mae = xgb.cv(dtrain=dmatrix,
                                num_boost_round=1000,
                                params=untuned_params,
                                nfold=5,
                                metrics="mae",
                                as_pandas=True,
                                seed=123)



# print("Untuned rmse: %f" % ((untuned_cv_results_rmse["test-rmse-mean"]).tail(1)))



# results_df = pd.DataFrame(gsearch.cv_results_)
# results_df.drop(
#     columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params'],
#                 inplace=True)
#
# results_df.sort_values(ascending=True, by=['rank_test_score'], axis=0, inplace=True)
# results_df.reset_index(inplace=True, drop=True)
#
# results_df = results_df.astype(float)
#
# # append predictions from the whole dataset back to the original info_df
# info_df['preds'] = gsearch.best_estimator_.predict(X)
# info_df['abs_error'] = abs(info_df['Total Revenue'] - info_df['preds'])
#
# info_df['Total Revenue_f'] = info_df['Total Revenue'].apply(lambda x: "$" + "{:,}".format(round(x)))
# info_df['preds_f'] = info_df['preds'].apply(lambda x: "$" + "{:,}".format(round(x)))
# info_df['abs_error_f'] = info_df['abs_error'].apply(lambda x: "$" + "{:,}".format(round(x)))
# info_df['predicted_rev_index'] = round((info_df['preds'] / info_df['terr_mean']) * 100, 2)
# info_df['abs_index_error'] = round(abs(info_df['rev_index'] - info_df['predicted_rev_index']), 2)
# info_df['index_error'] = round(info_df['predicted_rev_index'] - info_df['rev_index'], 2)
#
#
# output_df = info_df.loc[:, ['ns_id', 'CASS_Address', 'CASS_State', 'Territory', 'Total Revenue_f',
#                             'preds_f', 'abs_error_f', 'rev_index', 'predicted_rev_index', 'abs_index_error', 'index_error']]
# output_df.sort_values(by='abs_index_error', ascending=False, inplace=True)
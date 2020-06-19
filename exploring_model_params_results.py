import pandas as pd
from ast import literal_eval



results = pd.read_csv(r'C:\Users\nick\PycharmProjects\ss_xgb\models\grid_model_outputs\87b2ba90ac\87b2ba90ac.csv',
                      usecols=['param_colsample_bytree',
                               'param_gamma',
                               'param_learning_rate',
                               'param_max_depth',
                               'param_min_child_weight',
                               'param_n_estimators',
                               'param_reg_alpha',
                               'param_reg_lambda',
                               'param_subsample',
                               'rank_test_score',
                               'params'])

results.columns = [i.split('param_')[-1] for i in results.columns.tolist()]
best_params = literal_eval(results.iloc[0]['params'])

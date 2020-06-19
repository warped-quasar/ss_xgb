import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
from itertools import combinations
import itertools


# # will open features importance plots in new window
# matplotlib.use('TkAgg')


def get_n_param_combinations(param_dict):
    # param_dict = {'n_estimators': [50, 100, 150, 200, 250, 300, 350],
    #               "learning_rate": [0.05, 0.10, 0.25],
    #               "max_depth": [3, 5, 7, 9],
    #               "min_child_weight": [1, 2, 3, 4, 5, 6, 7],
    #               "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    #               "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #               "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #               'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}
    param_list_of_list = param_dict.values()
    res = list(itertools.product(*param_list_of_list))
    print(len(res))



def write_comparisons(df_w_preds, output_fp):
    acts_preds = list(zip(df_w_preds['act'], df_w_preds['preds'], df_w_preds['trev_preds']))
    with open(output_fp, 'w') as f:
        for act, preds, trev_pred in acts_preds[:5]:
            print('||||||||||||||||||||||||||||||||||||||||||||\n\n')
            f.write(f'Actual Revenue = ${round(act)}\n')
            f.write(f'Nick Predicted Revenue = ${round(preds)}\n')
            f.write(f'Trevor Predicted Revenue = ${round(trev_pred)}\n')
            f.write(f'Nick absolute difference: ${round(abs(preds - act))}\n')
            f.write(f'Trevor absolute difference: ${round(abs(trev_pred - act))}\n')
            f.write(f'Nick/Trev distance: ${round(abs(round(abs(trev_pred - act)) - round(abs(preds - act))))}\n')


def print_comparisons(df_w_preds):
    df_w_preds['nick_ab_diff'] = round(abs(df_w_preds['preds'] - df_w_preds['act']))
    df_w_preds['trev_ab_diff'] = round(abs(df_w_preds['trev_preds'] - df_w_preds['act']))

    print(f"\nNick mean error ${df_w_preds['nick_ab_diff'].mean()}")
    df_w_preds['nick_ab_diff'].hist()
    print(f"\nTrevor mean error ${df_w_preds['trev_preds'].mean()}")
    df_w_preds['trev_ab_diff'].hist()

    acts_preds = list(zip(df_w_preds['act'], df_w_preds['preds'], df_w_preds['trev_preds']))
    print('\n')
    for act, preds, trev_pred in acts_preds[:10]:
        print('||||||||||||||||||||||||||||||||||||||||||||')
        print(f'Actual Revenue = ${round(act)}')
        print(f'Nick Predicted Revenue = ${round(preds)}')
        print(f'Trevor absolute difference: ${round(abs(trev_pred - act))}')
        print(f'Nick absolute difference is: ${round(abs(preds - act))}')
        print(f'Trevor absolute difference is: ${round(abs(trev_pred - act))}')
        print(f'Nick/Trev distance ${round(abs(round(abs(trev_pred - act)) - round(abs(preds - act))))}')
        print('||||||||||||||||||||||||||||||||||||||||||||')

def plot_importances(model):
    xgb.plot_importance(model)

def plot_result_params(results_df, param_to_plot, num_to_plot):
    param_string = 'param_' + param_to_plot
    results_df[param_string][:num_to_plot].astype(float).plot(kind='bar', use_index=True, title=param_to_plot)
    plt.show()

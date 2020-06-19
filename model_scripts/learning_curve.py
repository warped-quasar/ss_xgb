from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
plt.style.use('seaborn')

from utils.master_datain import get_df_Xy

# will open features importance plots in new window
# matplotlib.use('TkAgg')

df, (X, y) = get_df_Xy(dataset='master_2020_renamed',
                       drop_unimportant=False,
                       drop_arcs=True)

learning_rate = 0.02529

xreg = xgb.XGBRegressor(objective='reg:squarederror',
                        # n_estimators=100,
                        # learning_rate=learning_rate,
                        # max_depth=9,
                        # min_child_weight=16
                        )

# xreg = xgb.XGBRegressor(objective='reg:squarederror',
#                         colsample_bytree=params['colsample_bytree'],
#                         gamma=params['gamma'],
#                         learning_rate=params['learning_rate'],
#                         max_depth=params['max_depth'],
#                         min_child_weight=params['min_child_weight'],
#                         # n_estimators=params['n_estimators'],
#                         subsample=params['subsample'],
#                         # reg_alpha=params['reg_alpha'],
#                         reg_lambda=params['reg_lambda'],
#                         seed=42,
#                         )

training_sizes = [1] + [int(round(x*len(X))) for x in np.linspace(0.1, 0.79, 20)]

# ### Bundling our previous work into a function ###
# def learning_curves(estimator, features, target, train_sizes, cv):

train_sizes, train_scores, validation_scores = learning_curve(estimator=xreg,
                                                              X=X,
                                                              y=y,
                                                              train_sizes=training_sizes,
                                                              cv=5,
                                                              scoring='neg_mean_absolute_error',
                                                              # shuffle=True,
                                                              n_jobs=-1)
train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, validation_scores_mean, label='Validation error')

plt.ylabel('MAE', fontsize=14)
plt.xlabel('Training set size', fontsize=14)
title = f'Learning curves: "XGB Reg" BASE MODEL'
# title = f'Learning curves: MD=12, MCW=9, ntree=100, eta=0.0225'

plt.title(title, fontsize=18, y=1.03)
plt.legend()
# plt.ylim(0, 100000)
plt.show()

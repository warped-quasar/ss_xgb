from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import explained_variance_score
import xgboost as xgb
import numpy as np
from utils.master_datain import get_df_Xy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
#
# # will open features importance plots in new window
matplotlib.use('TkAgg')

df, (X, y) = get_df_Xy(dataset='master_2020_renamed', drop_unimportant=True)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)


reg = LinearRegression().fit(X_train, y_train)
preds = reg.predict(X_test)

explained_variance = explained_variance_score(y_test, preds)

reg.score(X_test, y_test)
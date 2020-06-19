from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from utils.data_in import get_df_Xy
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb


df, (X, y) = get_df_Xy(dataset='reduced_feats_adj')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# "Learn" the mean from the training data
mean_train = np.mean(y_train)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))


params = {
          # default is 6, larger max_depth -> more complex (higher prob of overfitting)
          'max_depth': 4,
          # default is 1, larger min_child_weight -> more conservative
          'min_child_weight': 1,
          # default value 0.03, lower learning rate means more conservative
          'eta': .03,
          # default is 0, larger gamma -> more conservative
          'gamma': 0,
          # subsample ratio of columns when tree is constructed
          'colsample_bytree': 0.5,
          # default is 1, L2 regularization of weights, increasing -> more conservative
          'reg_lambda': 1,
          # default is 0, L1 regularization of weights, increasing -> more conservative
          'reg_alpha': 0,
          # used for regression, use 'reg:squarederror' because 'reg:linear' deprecated
          'objective': 'reg:squarederror',
          # use 'mae' or 'rmse', mae is in same units as revenue
          'eval_metric': 'mae'
}

# params = {
#     # Parameters that we are going to tune.
#     'max_depth': 5,
#     'min_child_weight': 9,
#     'eta': 0.0225,
#     'subsample': 0.7,
#     'colsample_bytree': 1,
#     'n_estimators':100
# }

learning_rate = 0.0225
xreg = xgb.XGBRegressor(objective='reg:squarederror',
                        colsample_bytree=params['colsample_bytree'],
                        # gamma=params['gamma'],
                        learning_rate=params['eta'],
                        max_depth=params['max_depth'],
                        min_child_weight=params['min_child_weight'],
                        n_estimators=10000,
                        # subsample=params['subsample'],
                        # reg_alpha=params['reg_alpha'],
                        # reg_lambda=params['reg_lambda'],
                        seed=42)


eval_set = [(X_test, y_test)]
xreg.fit(X_train, y_train,
         eval_metric="mae",
         eval_set=eval_set,
         early_stopping_rounds=200,
         verbose=True)

preds = xreg.predict(X_test)

# scores = r2_score(preds, y_test, multioutput='variance_weighted')

mae = mean_absolute_error(y_test, preds)
print("MAE: %f" % (mae))
print(f'ETA = {learning_rate}')

# df['preds'] = preds


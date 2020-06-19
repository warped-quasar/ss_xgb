import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import json
# will open features importance plots in new window
matplotlib.use('TkAgg')

from utils.master_datain import get_df_Xy
df, (X, y) = get_df_Xy(dataset='master_2020_renamed', drop_unimportant=True)

# # load JS visualization code to notebook
# shap.initjs()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    # random_state=123,
                                                    shuffle=True
                                                    )


# params = {
#         'colsample_bytree': 1,
#           'gamma': 0,
#           'eta': 0.0225,
#           'max_depth': 3,
#           'min_child_weight': 12,
#           'n_estimators': 100,
#           # 'reg_alpha': 0.75,
#           # 'reg_lambda': 0.45,
#           'subsample': 1}

# eta = 0.025297
# # for baseline results use empty xreg below
xreg = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.0225,
    max_depth=9,
    min_child_weight=16,
    # subsample=0.6,
    # colsample_bytree=0.6,
    # reg_alpha=0.75,
    # reg_lambda=0.45,
)

xreg.fit(X, y)

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(xreg)
shap_values = explainer.shap_values(X)
# summarize the effects of all the features
shap.summary_plot(shap_values, X)




"""Create SHAP df"""
shap_df = pd.DataFrame(shap_values)

col_names = X.columns.tolist()
shap_df.columns = col_names

# get abs mean of SHAP values to create SHAP feature imps list
shap_imps = shap_df.abs().mean()
shap_imps.sort_values(ascending=False, inplace=True)

# removing all vars that don't have any SHAP weight
not_important_list = shap_imps[shap_imps == 0].index.tolist()
not_important_dict = {'not_important' : not_important_list}

# # store least important vars in a json file
# with open(r'C:\Users\nick\PycharmProjects\ss_xgb\unimportant_var_jsons\base_not_important_vars.json', 'w') as ni_json:
#     json.dump(not_important_dict, ni_json, sort_keys=True, indent=4)





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

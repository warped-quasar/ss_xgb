from model_scripts.run_single_regressor import run_one_regressor
from model_scripts.compare_model_shap_imps import get_shap_comparisons
from utils.data_in import get_df_Xy



params = {'colsample_bytree': 0.7,
          'gamma': 0,
          'learning_rate': 0.0225,
          'max_depth': 3,
          'min_child_weight': 12,
          'n_estimators': 100,
          'reg_alpha': 0.75,
          'reg_lambda': 0.45,
          'subsample': 1}

dataset = 'reduced_feats_adj'

df, (X, y) = get_df_Xy(dataset)

# params = 'default'
# TODO: need to fix n_estimators to be dynamic
early_stopping = False

one_regressor_res_df = run_one_regressor(df, X, y, params=params, early_stopping=early_stopping)
# {'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 50}

shap_feats_df_base = get_shap_comparisons(df, X, y, params=params)

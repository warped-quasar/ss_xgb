import xgboost as xgb
import shap
from utils.data_in import get_df_Xy
import matplotlib
import numpy as np
import sqlite3
import pandas as pd
import matplotlib

# will open features importance plots in new window
matplotlib.use('TkAgg')

# load JS visualization code to notebook
shap.initjs()

df, (X, y) = get_df_Xy('original revenue')

X = X[['pop_sqmi',
       'Apparel Stores',
       'Thrift Stores',
       'Total % Moved from abroad',
       'Total % Moved from different county within same state',
       'Total % Moved from different state',
       'Total % Moved within same county',
       'Total % Women who had a birth in the past 12 months',
       'Afghan',
       'Albanian',
       'American',
       'Arabian',
       'Armenian',
       'Austrian',
       'Canadian',
       'Cypriot',
       'Czechoslovakian',
       'French (except Basque)',
       'French Canadian',
       'German',
       'German Russian',
       'Greek',
       'Iranian',
       'Irish',
       'Italian',
       'Macedonian',
       'Pennsylvania German',
       'Polish',
       'Portuguese',
       'Russian',
       'Scotch-Irish',
       'Scottish',
       'Slovak',
       'Slovene',
       'Subsaharan African',
       'Yugoslavian',
       'Total (Pop) % Age 55+',
       'Total (Pop) % Age   5-9',
       'Total (Pop) % Age 18-19',
       'Total (Pop) % Age 25-29',
       'Total (Pop) % Age 35-39',
       'Total (Pop) % Age 40-44',
       'Total (Pop) % Age 65',
       'Total (Pop):Detail % Age  4',
       'Total (Pop):Detail % Age 17',
       'Total % No schooling completed',
       'Total % 9th grade',
       'Total % 11th grade',
       'Total % Some college less than 1 year',
       'Total % Some college 1 or more years no degree',
       "Total % Associate's degree",
       'Total (HHs) % 3-Person Households',
       'Total (HHs) % 4-Person Households',
       'Household Income  $10000-$14999',
       'Household Income  $25000-$29999',
       'Household Income  $30000-$34999',
       'Household Income  $60000-$74999',
       'Household Income $125000-$149999',
       'Total % Age 18+',
       'Race Black Below Poverty Level',
       'Ratio of Income-Poverty Level: 1.50-1.84',
       'Ratio of Income-Poverty Level: 1.85-1.99',
       'Employees: NAICS by Business Sector (2 digit code) % Utilities (22)',
       'Employees: NAICS by Business Sector (2 digit code) % Construction (23)',
       'Employees: NAICS by Business Sector (2 digit code) % Manufacturing (31)',
       'Employees: NAICS by Business Sector (2 digit code) % Manufacturing (32)',
       'Employees: NAICS by Business Sector (2 digit code) % Manufacturing (33)',
       'Employees: NAICS by Business Sector (2 digit code) % Wholesale Trade (42)',
       'Employees: NAICS by Business Sector (2 digit code) % Retail Trade (45)',
       'Employees: NAICS by Business Sector (2 digit code) % Transportation and Warehousing (49)',
       'Employees: NAICS by Business Sector (2 digit code) % Finance and Insurance (52)',
       'Employees: NAICS by Business Sector (2 digit code) % Professional Scientific and Technical Services (54)',
       'Employees: NAICS by Business Sector (2 digit code) % Administrative and Support and Waste Management and Remediation Services (56)',
       'Employees: NAICS by Business Sector (2 digit code) % Educational Services (61)',
       'Employees: NAICS by Business Sector (2 digit code) % Health Care and Social Assistance (62)',
       'Employees: NAICS by Business Sector (2 digit code) % Arts Entertainment and Recreation (71)',
       'Employees: NAICS by Business Sector (2 digit code) % Accommodation and Food Services (72)',

       'Establishments: NAICS by Business Sector (2 digit code) % Agriculture Forestry Fishing and Hunting (11)',
       'Establishments: NAICS by Business Sector (2 digit code) % Mining Quarrying and Oil and Gas Extraction (21)',
       'Establishments: NAICS by Business Sector (2 digit code) % Manufacturing (31)',
       'Establishments: NAICS by Business Sector (2 digit code) % Wholesale Trade (42)',
       'Establishments: NAICS by Business Sector (2 digit code) % Transportation and Warehousing (48)',
       'Establishments: NAICS by Business Sector (2 digit code) % Transportation and Warehousing (49)',
       'Establishments: NAICS by Business Sector (2 digit code) % Information (51)',
       'Establishments: NAICS by Business Sector (2 digit code) % Management of Companies and Enterprises (55)',
       'Establishments: NAICS by Business Sector (2 digit code) % Educational Services (61)',
       'Establishments: NAICS by Business Sector (2 digit code) % Health Care and Social Assistance (62)',
       'Establishments: NAICS by Business Sector (2 digit code) % Accommodation and Food Services (72)',
       'Establishments: NAICS by Business Sector (2 digit code) % Other Services (except Public Administration) (81)',
       'Establishments: NAICS by Business Sector (2 digit code) % Public Administration (92)',
       'Housing Value     $10000',
       'Housing Value   $15000-$19999',
       'Housing Value   $20000-$24999',
       'Housing Value   $30000-$34999',
       'Housing Value   $80000-$89999',
       'Housing Value   $90000-$99999',
       'Housing Value  $125000-$149999',
       'Housing Value  $150000-$174999',
       'Housing Value  $175000-$199999',
       'Housing Value  $200000-$249999',
       'Housing Value $1000000 or more',
       'B Flourishing Families % B07 Generational Soup',
       'B Flourishing Families % B09 Family Fun-tastic',
       'D Suburban Style % D15 Sports Utility Families',
       'D Suburban Style % D16 Settled in Suburbia',
       'H Middle-class Melting Pot % H26 Progressive Potpourri',
       'H Middle-class Melting Pot % H29 Destination Recreation',
       'I Family Union % I31 Blue Collar Comfort',
       'K Significant Singles % K40 Bohemian Groove',
       'N Pastoral Pride % N48 Rural Southern Bliss',
       'O Singles and Starters % O50 Full Steam Ahead',
       'O Singles and Starters % O52 Urban Ambition',
       'P Cultural Connections % P58 Heritage Heights',
       'P Cultural Connections % P60 Striving Forward',
       'S Economic Challenges % S71 Tough Times',
       'Total:Details:Agriculture forestery fishing/hunting mining % Agriculture forestry fishing/hunting',
       'Total:Details:Professional sci mgmt admin and waste mgmt svcs % Mgmt of companies and enterprises',
       'Total:Details:Educational svcs health care and social asst % Educational svcs',
       'Total:Details:Arts entertainmnt recreation accom. and food svcs % Arts entertainment recreation',
       'Total:Details:Arts entertainmnt recreation accom. and food svcs % Accommodation and food svcs',
       'Construction',
       'Manufacturing',
       'Wholesale trade',
       'Information',
       'Finance and ins real estate rental and leasing',
       'Professional sci mgmt admin and waste mgmt svcs',
       'Furniture & Home Furnishings Stores',
       'Building Material & Garden Equipment & Supply Dealers',
       'Clothing & Clothing Accessories Stores',
       'Food Services and Drinking Places',
       'General Merchandise Apparel and Accessories Furniture and Other Sales',
       'Motor Vehicle & Parts Dealers Other Motor Vehicle Dealers',
       'Furniture & Home Furnishings Stores Total Home Furnishing Stores',
       'Furniture & Home Furnishings Stores:Home Furnishing Stores Other Home Furnishings Stores',
       'Building Material & Garden Equipment & Supply Dealers:Lawn and Garden Equipment and Supplies Stores Nursery and Garden centers',
       'Food and Beverage Stores Total Grocery Stores',
       'Food and Beverage Stores Beer Wine & Liquor Stores',
       "Clothing & Clothing Accessories Stores:Clothing Stores Children's and Infants' Clothing Stores",
       'Clothing & Clothing Accessories Stores Total Jewelry Luggage & Leather Goods Stores',

       'Sporting Goods Hobby Book & Music Stores:Sporting Goods Hobby & Musical Instrument Stores Sporting Goods Stores',
       'Sporting Goods Hobby Book & Music Stores:Sporting Goods Hobby & Musical Instrument Stores Sew/Needlework/Piece Goods Stores',

       'Sporting Goods Hobby Book & Music Stores:Book Periodical & Music Stores Total Book Stores and News Dealers',
       'Sporting Goods Hobby Book & Music Stores:Book Periodical & Music Stores:Book Stores and News Dealers Book Stores',
       'Food Services and Drinking Places Restaurants and other Eating Places',
       'CASS_StateKS',
       'trev_preds',
       'act']]

# params = {'colsample_bytree': 0.5,
#           'gamma': 0,
#           'learning_rate': 0.05,
#           'max_depth': 4,
#           'min_child_weight': 1,
#           'n_estimators': 143,
#           # 'reg_alpha': 1e-05,
#           # 'reg_lambda': 1e-05,
#           # 'subsample': 0.95
#           }

# params = {'subsample': 0.6,
#           'reg_lambda': 10,
#           'min_child_weight': 6,
#           'max_depth': 8,
#           'learning_rate': 0.1,
#           'gamma': 0.0,
#           'colsample_bytree': 0.8}

# {'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 1,
#  'n_estimators': 50}

# # reg:linear is deprecated, use 'reg:squarederror' for regression
# xreg = xgb.XGBRegressor(objective='reg:squarederror',
#                         colsample_bytree=params['colsample_bytree'],
#                         gamma=params['gamma'],
#                         learning_rate=params['learning_rate'],
#                         max_depth=params['max_depth'],
#                         min_child_weight=params['min_child_weight'],
#                         n_estimators=params['n_estimators'],
#                         # subsample=params['subsample'],
#                         # reg_alpha=params['reg_alpha'],
#                         # reg_lambda=params['reg_lambda'],
#                         seed=42)
xreg = xgb.XGBRegressor()
# reg:linear is deprecated, use 'reg:squarederror' for regression
# xreg = xgb.XGBRegressor(objective='reg:squarederror',
#                         colsample_bytree=params['colsample_bytree'],
#                         # gamma=params['gamma'],
#                         learning_rate=params['learning_rate'],
#                         max_depth=params['max_depth'],
#                         min_child_weight=params['min_child_weight'],
#                         n_estimators=params['n_estimators'],
#                         # subsample=params['subsample'],
#                         # reg_alpha=params['reg_alpha'],
#                         # reg_lambda=params['reg_lambda'],
#                         seed=42)

xreg.fit(X, y)

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(xreg)
shap_values = explainer.shap_values(X)

df['preds'] = xreg.predict(X)
df['abs_error'] = abs(df['act'] - df['preds'])
df['abs_error_f'] = df['abs_error'].apply(lambda x: "$" + "{:,}".format(round(x)))
df['act_f'] = df['act'].apply(lambda x: "$" + "{:,}".format(round(x)))
df['preds_f'] = df['preds'].apply(lambda x: "$" + "{:,}".format(round(x)))

# summarize the effects of all the features
shap.summary_plot(shap_values, X)

# # for i in range(0,5):
# #     print(f'Plot {i + 1} data:  {X.iloc[i,:]}')
# #     # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[17,:], X.iloc[17,:], matplotlib=True,
#                 figsize=(20,5), text_rotation=45)
#
# shap.dependence_plot("p", shap_values, X)

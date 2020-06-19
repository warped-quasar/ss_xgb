import sqlite3
import pandas as pd
from all_var_names import all_var_names

conn = sqlite3.connect(r"C:\Users\nick\PycharmProjects\ss_xgb\model_2020_data\model_data_2020.sqlite")
columns_df = pd.read_sql_query("SELECT * FROM colnames_ids", conn)
columns_df.to_csv(r'colnames_colids_cleaned_colnames.csv', index=False)

new_var_names = []
for i in all_var_names:
    cleaned_i = i.replace(',', '')
    cleaned_i = cleaned_i.replace("'", '')
    cleaned_i = cleaned_i.replace("/", '&')
    cleaned_i = cleaned_i.replace('+', 'plus')
    cleaned_i = cleaned_i.replace(':', '-')
    new_var_names.append(cleaned_i)

columns_df['cleaned_var_names'] = new_var_names

columns_df.to_sql('colnames_ids', conn, if_exists='replace', index=False)

new_var_list = columns_df['cleaned_var_names'].tolist()
new_var_set_list = list(set(new_var_list))
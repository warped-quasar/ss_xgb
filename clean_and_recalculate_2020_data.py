import pandas as pd
from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
sns.set(style="ticks", color_codes=True)


arc_all_locs_w_present = r'C:\Users\nick\PycharmProjects\ss_xgb\data\2020 data from Jake\2020 Data Cleaned_zips_cleaned.csv'
df = pd.read_csv(arc_all_locs_w_present,
                 dtype={'Store': str,
                        'Store Present %': float,
                        'CASS_ZIP':str,
                        'CASS_Plus4':str}
                 )


df_2020 = df.copy()

# southern only has 11 months of data, subtracting 1 month to all southern locations
df_2020.loc[df_2020['Territory'] == 'Southern', 'Store Present %'] = df_2020.loc[df_2020['Territory'] == 'Southern', 'Store Present %'] - 100 *(1 - 11/12)


def filter_least_present_locations(df_to_filter_presents, percent_threshold):
    df_to_filter_presents = df_to_filter_presents.loc[df_to_filter_presents['Store Present %'] >= percent_threshold]
    return df_to_filter_presents

df_2020 = filter_least_present_locations(df_to_filter_presents=df_2020, percent_threshold=80)

df_2020['adjusted_revenue'] = df_2020['Total Revenue'] * (1 + ((100 - (df_2020['Store Present %']))/100))


df_ids = [uuid4().hex[:10] for _ in range(len(df_2020))]
df_2020.insert(loc=0, column='model_ids', value=df_ids)

conn = sqlite3.connect(r'C:\Users\nick\PycharmProjects\ss_xgb\model_data_2020.sqlite')
c = conn.cursor()

df_2020.to_sql('modeling_data_facts', conn, if_exists='fail', index=False)
# # plot adjusted vs actual
# sns.catplot(x="Territory", y="Total Revenue", kind='box', data=df_2020)
# _ = plt.title('Total Revenue')
# plt.show()
#
# # plot adjusted vs actual
# sns.catplot(x="Territory", y="adjusted_revenue", kind='box', data=df_2020)
# _ = plt.title('Adjusted Revenue')
# plt.show()


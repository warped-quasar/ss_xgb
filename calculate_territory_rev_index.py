import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# will open features importance plots in new window
# matplotlib.use('TkAgg')

con = sqlite3.connect(r"C:\Users\nick\PycharmProjects\ss_xgb\model_2020_data\model_data_2020.sqlite")
model_facts = pd.read_sql('SELECT * FROM model_facts_2020_master', con)
# model_data = pd.read_sql('SELECT * FROM modeling_2020', con)



# groupby_territory
model_factz = model_facts.copy()
# territory names strings = ['Central', 'Eastern', 'Southern', 'Western']
central = model_factz.loc[model_factz['Territory'] == 'Central']
eastern = model_factz.loc[model_factz['Territory'] == 'Eastern']
southern = model_factz.loc[model_factz['Territory'] == 'Southern']
western = model_factz.loc[model_factz['Territory'] == 'Western']

print(len(model_factz))
print(len(central), len(eastern), len(southern), len(western))
assert len(model_facts) == len(model_factz) == sum([len(central), len(eastern), len(southern), len(western)])

# create indexes
central_mean = model_factz.loc[model_factz['Territory'] == 'Central']['Total Revenue'].mean()
eastern_mean = model_factz.loc[model_factz['Territory'] == 'Eastern']['Total Revenue'].mean()
southern_mean = model_factz.loc[model_factz['Territory'] == 'Southern']['Total Revenue'].mean()
western_mean = model_factz.loc[model_factz['Territory'] == 'Western']['Total Revenue'].mean()

mean_rev_mapping_dict = {'Central': central_mean,
                         'Eastern': eastern_mean,
                         'Southern': southern_mean,
                         'Western': western_mean}

model_factz['terr_mean'] = model_factz['Territory'].map(mean_rev_mapping_dict)

model_factz['rev_index'] = (model_factz['Total Revenue'] / model_factz['terr_mean']) * 100

model_factz.to_sql('model_facts_2020_master', con, if_exists='replace', index=False)
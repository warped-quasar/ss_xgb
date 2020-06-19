import pandas as pd
import sqlite3
import json


# with open(r'C:\Users\nick\PycharmProjects\ss_xgb\unimportant_var_jsons\base_not_important_vars.json', 'r') as ni_json:
#     data = json.load(ni_json)
#     not_important_vars = data['not_important']
#
# arc_flag_true = ['d6d60b79_78bb',
#  '76a64e73_d8e2',
#  'd91c7823_294f',
#  '3db61ba3_f007',
#  '76b32d51_c10c',
#  'b431a885_414b',
#  '6084e84c_4e98',
#  'e5e3f025_6cf4',
#  '72dd9118_076e',
#  '1f6dc42b_2c07',
#  'e58a309f_21d8',
#  '9d9a108d_d848',
#  'c143762e_1808',
#  'e584b2df_44ba',
#  'bc2b882e_7a12']

def get_arc_flags(master_df):
    marked_arcs = pd.read_csv(r'C:\Users\nick\PycharmProjects\ss_xgb\MASTER_DATA\MARKING ARCs - Final Output.csv')
    marked_arcs = marked_arcs[['CASS_Address', 'arc_flag']]

    complete = pd.merge(left=master_df, right=marked_arcs, left_on='CASS_Address', right_on='CASS_Address', how='left',
                        indicator=True)


def new_get_df_Xy(drop_arcs):
    master_df = pd.read_csv(
        r'C:\Users\nick\PycharmProjects\ss_xgb\MASTER_DATA\MARKING ARCs - Amending Master Data.csv'

    )
    master_df['CASS_ZIP'] = master_df['CASS_ZIP'].astype(str).str.zfill(5)

    if drop_arcs:
        master_df = master_df.loc[master_df['arc_flag'] == 'True']



def get_df_Xy(dataset, drop_unimportant, drop_arcs):
    dataset = 'master_2020_renamed'
    con = sqlite3.connect(r'C:\Users\nick\PycharmProjects\ss_xgb\MASTER_DATA\master_2020.sqlite')
    master_df = pd.read_sql(f'SELECT * FROM {dataset}', con=con)

    if drop_arcs:
        master_df = master_df.loc[~master_df['ns_id'].isin(arc_flag_true)]
    elif drop_arcs is False:
        master_df = master_df
    else:
        pass

    # remove family store that is right next to ARC
    master_df.drop([10], inplace=True)
    # add the dropped row's revenue to the ARC
    master_df.iat[11, -5] = 1335310.8443
    master_df.reset_index(inplace=True, drop=True)

    df = master_df.copy()

    # if drop_unimportant is True:
    #     df.drop(columns=not_important_vars, inplace=True)
    # elif drop_unimportant is False:
    #     df = df
    # else:
    #     pass

    non_data_columns = ['CASS_Address',
                        'ARC',
                        'Store',
                        'CASS_City',
                        'CASS_State',
                        'CASS_ZIP',
                        'CASS_Plus4',
                        'Territory',
                        'Store Present pct',
                        'terr_mean',
                        'rev_index',
                        'ns_id',
                        'Right_Territory',
                        'Username',
                        'Location ID',
                        'image_path',
                        ]

    unwanted_data_cols = ['ests_pct Other Services (except Public Administration) (81)',
                          'emps_pct Other Services (except Public Administration) (81)']

    df.drop(columns=unwanted_data_cols, inplace=True)
    return_df = df.copy()

    df.drop(columns=non_data_columns, inplace=True)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return return_df, (X, y)

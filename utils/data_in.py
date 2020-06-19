import sqlite3
import pandas as pd

con = sqlite3.connect(r"C:\Users\nick\PycharmProjects\ss_xgb\model_2020_data\model_data_2020.sqlite")
model_df = pd.read_sql_query("SELECT * from modeling_2020", con)
model_facts = pd.read_sql('SELECT * FROM model_facts_2020_master', con)

friendly_model_facts = model_facts[['model_ids',
                                    'CASS_Address',
                                    'CASS_City',
                                    'CASS_State',
                                    'CASS_ZIP',
                                    'Territory',
                                    'Total Revenue',
                                    'rev_index']]

columns_df = pd.read_sql_query("SELECT * FROM colnames_ids", con)
reduced_features = pd.read_sql_query("SELECT * FROM new_reduced_vars", con)['new_reduced_vars'].tolist()
no_business = pd.read_sql_query("SELECT * FROM no_business_vars", con)['No Business'].tolist()
    # .extend(['Right_Territory','Username', 'Location ID', 'image_path', 'model_ids','Total Revenue'])


colnames_replace = dict(zip(columns_df['Column Names'], columns_df['cleaned_var_names']))
vars_to_drop = ['Right_Territory',
                'Username',
                'Location ID',
                'image_path',
                'est_Other Services (except Public Administration) (81)',
                'emps_Other Services (except Public Administration) (81)']


def get_df_Xy(dataset):
    if dataset == 'original_revenue':
        original_df = model_df.copy()
        colnames_replace['Total Revenue'] = 'act'
        original_df.rename(columns=colnames_replace, inplace=True)
        original_df.drop(columns=vars_to_drop, inplace=True)

        # original_df.rename(columns=colnames_replace, inplace=True)

        X, y = original_df.iloc[:, :-1], original_df.iloc[:, -1]
        X.drop(columns=['model_ids'], inplace=True)

        original_df = pd.merge(original_df, friendly_model_facts, on='model_ids')
        return original_df, (X, y)

    if dataset == 'adjusted_revenue':
        adjusted_df = model_df.copy()
        colnames_replace['adjusted_revenue'] = 'act'
        adjusted_df.rename(columns=colnames_replace, inplace=True)
        adjusted_df.drop(columns=vars_to_drop, inplace=True)

        adjusted_df.drop(columns=['Total Revenue'], inplace=True)
        adjusted_df.rename(columns=colnames_replace, inplace=True)
        X, y = adjusted_df.iloc[:, :-1], adjusted_df.iloc[:, -1]
        X.drop(columns=['model_ids'], inplace=True)

        adjusted_df = pd.merge(adjusted_df, friendly_model_facts, on='model_ids')
        return adjusted_df, (X, y)

    if dataset == 'reduced_feats_adj':
        reduce_adj_df = model_df.copy()
        colnames_replace['adjusted_revenue'] = 'act'
        reduce_adj_df.rename(columns=colnames_replace, inplace=True)
        df_to_reduce = reduce_adj_df.copy()
        reduced_reduce_adj_df = df_to_reduce[reduced_features]

        vars_to_drop.remove('est_Other Services (except Public Administration) (81)')
        vars_to_drop.remove('emps_Other Services (except Public Administration) (81)')

        reduced_reduce_adj_df.drop(columns=vars_to_drop, inplace=True)
        reduced_reduce_adj_df.drop(columns=['Total Revenue'], inplace=True)
        X, y = reduced_reduce_adj_df.iloc[:, :-1], reduced_reduce_adj_df.iloc[:, -1]
        X.drop(columns=['model_ids'], inplace=True)

        reduced_reduce_adj_df = pd.merge(reduced_reduce_adj_df, friendly_model_facts, on='model_ids')
        return reduced_reduce_adj_df, (X, y)


    # if dataset == 'no_business_adj':
    #     no_biz_df = model_df.copy()
    #     colnames_replace['adjusted_revenue'] = 'act'
    #     no_biz_df.rename(columns=colnames_replace, inplace=True)
    #     df_to_reduce = no_biz_df.copy()
    #     reduced_no_biz_df = df_to_reduce[no_business]
    #     vars_to_drop.remove('est_Other Services (except Public Administration) (81)')
    #     vars_to_drop.remove('emps_Other Services (except Public Administration) (81)')
    #
    #     reduced_no_biz_df.drop(columns=vars_to_drop, inplace=True)
    #     reduced_no_biz_df.drop(columns=['Total Revenue'], inplace=True)
    #     X, y = reduced_no_biz_df.iloc[:, :-2], reduced_no_biz_df.iloc[:, -1]
    #     X.drop(columns=['model_ids'], inplace=True)
    #
    #     reduced_no_biz_df = pd.merge(reduced_no_biz_df, friendly_model_facts, on='model_ids')
    #     return reduced_no_biz_df, (X, y)

    if dataset == 'trevs_feats_adj':
        from utils.selecting_vars_04292020 import eq_trevs_vars
        trevs_feats_df = model_df.copy()
        colnames_replace['adjusted_revenue'] = 'act'
        trevs_feats_df = trevs_feats_df[eq_trevs_vars]
        trevs_feats_df.rename(columns=colnames_replace, inplace=True)
        trevs_feats_df.drop(columns=vars_to_drop, inplace=True)
        trevs_feats_df.drop(columns=['Total Revenue'], inplace=True)
        X, y = trevs_feats_df.iloc[:, :-2], trevs_feats_df.iloc[:, -1]
        X.drop(columns=['model_ids'], inplace=True)

        trevs_feats_df = pd.merge(trevs_feats_df, friendly_model_facts, on='model_ids')
        return trevs_feats_df, (X, y)
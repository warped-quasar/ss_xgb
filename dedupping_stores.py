from utils.master_datain import get_df_Xy

from collections import Counter

df, (X, y) = get_df_Xy(dataset='master_2020_renamed')

def multiple_zips(df):
    counter_zip = Counter(df['CASS_ZIP'])
    counter_zip.most_common(10)
    multiples_zip = []
    for item, count in counter_zip.most_common():
        if count > 1:
            print(item)
            multiples_zip.append(item)

    multiples_zip_df = df.loc[df['CASS_ZIP'].isin(multiples_zip)]
    return multiples_zip_df

def multiple_popsqmi(df):
    counter_pop_sqmi = Counter(df['pop_sqmi'])
    counter_pop_sqmi.most_common(10)
    multiples_popsqmi = []
    for item, count in counter_pop_sqmi.most_common():
        if count > 1:
            print(item)
            multiples_popsqmi.append(item)

    multiples_popsqmi_df = df.loc[df['pop_sqmi'].isin(multiples_popsqmi)]
    return multiples_popsqmi_df

multiples_zip_df = multiple_zips(df)
multiples_popsqmi_df = multiple_popsqmi(df)
import pandas as pd


def handle_data(dataset, redundant_cols, describe_cols):
    for i in dataset:
        for col in redundant_cols:
            i.drop(col, axis=1, inplace=True)
        i['Parking'] = i['Parking'].map(lambda x: None if x == '0 spaces' else x)
        for col in describe_cols:
            i[col] = i[col].map(f)
    return dataset[0].fillna(0), dataset[1].fillna(0)


def f(x):
    if pd.isnull(x):
        return 0
    else:
        count = str(x).count(',')
        if count < 1:
            return 1
        else:
            return count + 1
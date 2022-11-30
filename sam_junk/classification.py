import pandas as pd


def get_data(filepath):
    data = pd.read_csv(filepath, low_memory=False, decimal=',')
    data = drop_column(data, 'reviewTime')
    data = drop_column(data, 'asin')
    data = drop_column(data, 'style')
    data = drop_column(data, 'reviewTime')
    data = drop_column(data, 'asin')


def drop_column(data, col_name):
    return data.drop(col_name, axis=1, inplace=False)


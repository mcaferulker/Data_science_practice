from sklearn.ensemble import RandomForestRegressor

import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
import numpy as np

import math

import os

os.makedirs('tmp', exist_ok= True)


PATH = "./bluebook-for-bulldozers/"

df = pd.read_csv(f'{PATH}Train.csv', low_memory=False, parse_dates=["saledate"])

df = df.sort_values(by="saledate")

# Categorical Variables

def train_cats(df):         # Convert string dtype to pandas category type.
    for n,c in df.items():
        if is_string_dtype(c):
            df[n] = c.astype("category").cat.as_ordered()

def apply_cats(df, train):
    for n,c in df.items():
        if train[n].ntype == "category":
            df[n] = pd.Categorical(c, categories= train[n].cat.categories, ordered= True)

train_cats(df)


df["UsageBand"] = df["UsageBand"].cat.set_categories(["High", "Medium", "Low"], ordered= True)  # Since Future Warning, "No more 'Inplace Method'", set_categories used.

#print(df["UsageBand"].cat.codes)

def numericalize(df, col, name):
    if not is_numeric_dtype(col):
        df[name] = col.cat.codes + 1

numericalize(df, df["UsageBand"], "UsageBand")

#print(df["UsageBand"])

# DATE TIME COLUMN
def add_datepart(df, dt_name, drop=True):

    dt_column = df[dt_name]

    attr = ['year', 'month', 'week', 'dayofweek', 'dayofyear', 'is_month_end',
            'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start']

    for a in attr:
        if not a == "week":
            df["Date" + a.capitalize()] = getattr(dt_column.dt, a)
        else:
            df["Date" + a] = getattr(dt_column.dt.isocalendar(), a)
    df["Date" + "Elapsed"] = dt_column.astype(np.int64) // 10 ** 9

    if drop:
        df.drop(dt_name, axis =1, inplace = True)

add_datepart(df, "saledate")

# MISSING VALUES

def fix_missing_values(df, col, name, nan_dict, is_train):

    if is_train:
        if is_numeric_dtype(col):
            if pd.isnull(col).sum:
                df[name + "_nan"] = pd.isnull(col)
                nan_dict[name] = col.median()
                df[name] = col.fillna(col.median())

    else:
        if is_numeric_dtype(col):
            if name in nan_dict:
                df[name + "_nan"] = pd.isnull(col)
                df[name] = col.fillna(nan_dict[name])
            else:
                df[name] = col.fillna(df[name].median())


# Numeric and missing
def proc_df(df, y_field,nan_dict = None, is_train = True):
    df = df.copy()
    y = df[y_field].values
    df.drop(y_field,axis = 1, inplace = True)

    if nan_dict is None:
        nan_dict = {}

    for n, c in df.items():
        fix_missing_values(df, c, n, nan_dict, is_train)
        numericalize(df, c, n)

    if is_train:
        return df, y, nan_dict

    return df, y

def split_train_val(df, n):
    return df[:n].copy(), df[n:].copy()

n_valid = 12000 # same as kaggle's test set size
n_train = len(df) - n_valid
raw_train, raw_valid = split_train_val(df, n_train)

x_train, y_train, nas = proc_df(raw_train, 'SalePrice')
x_valid, y_valid = proc_df(raw_valid, 'SalePrice', nan_dict= nas, is_train=False)


# First Model

m = RandomForestRegressor(n_estimators=10, n_jobs=-1, max_features= 0.5)
m.fit(x_train,y_train)


def rmse(x, y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    print(f"RMSE of train set {rmse(m.predict(x_train), y_train)}")
    print(f"RMSE of validation set {rmse(m.predict(x_valid), y_valid)}")
    print(f"R^2 of train set {m.score(x_train, y_train)}")
    print(f"R^2 of validation set {m.score(x_valid, y_valid)}")

print_score(m)
print('---------------------------------------')

# Check important features
feature_importances_df = pd.DataFrame(
    {"feature": list(x_train.columns), "importance": m.feature_importances_}
).sort_values("importance", ascending=False)

# Display
print(feature_importances_df)

""" 
# Drop less important features
df.drop(list(feature_importances_df.loc[feature_importances_df["importance"] < 0.06, 'feature']), axis=1)
# After editing the data, the model can be re-run again. It's not done here.
"""
# Subset Create
def get_sample(df, n):

    idxs = np.random.permutation(len(df))[:n]
    return idxs, df.iloc[idxs].copy()

idxs, x_train_dp = get_sample(x_train, 3000)
y_train_dp = y_train[idxs]

m.fit(x_train,y_train)
print_score(m)





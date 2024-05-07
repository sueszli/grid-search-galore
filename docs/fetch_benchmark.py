from timeit import default_timer as timer

import openml
from ucimlrepo import fetch_ucirepo

import pandas as pd
from scipy.io.arff import loadarff
from pathlib import Path


def timer_func(func):
    def wrapper(*args, **kwargs):
        t1 = timer()
        result = func(*args, **kwargs)
        t2 = timer()
        print(f"{func.__name__}() executed in {(t2-t1):.6f}s")
        return result

    return wrapper


@timer_func
def fetch_seattle_crime_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    # see: https://www.openml.org/search?type=data&id=41960&sort=runs&status=active
    # see: https://openml.github.io/openml-python/develop/examples/30_extended/datasets_tutorial.html#download-datasets

    dataset = openml.datasets.get_dataset(41960, download_data=True, download_qualities=True, download_features_meta_data=True)

    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    y = pd.DataFrame(y, columns=[dataset.default_target_attribute])
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    return X, y


@timer_func
def fetch_seattle_crime_dataset_offline() -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = Path(__file__).resolve().parent / "data" / "seattle.arff"
    data = loadarff(file_path)
    df = pd.DataFrame(data[0])

    # unclear why it's missing in the online version
    df = df.drop("Report_Number", axis=1)

    # convert bytes to str
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    y = pd.DataFrame(df["Primary_Offense_Description"])  # pd.Series to df
    X = df.drop("Primary_Offense_Description", axis=1)
    return X, y


def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    if df1.equals(df2):
        print("df1 and df2 are both identical, nothing to check")
        return

    shape_matching = df1.shape == df2.shape
    if not shape_matching:
        print("shapes do not match")
        print("df1.shape:", df1.shape)
        print("df2.shape:", df2.shape)
        print("df1.index:\n", df1.index)
        print("df2.index:\n", df2.index)
        print("\n" * 2)

    only_in_df1 = set(df1.columns) - set(df2.columns)
    only_in_df2 = set(df2.columns) - set(df1.columns)
    column_matching = df1.columns.equals(df2.columns)
    if not column_matching:
        print("columns do not match")
        diff_dict = {
            "column name": list(only_in_df1) + list(only_in_df2),
            "in df1": [col in df1.columns for col in list(only_in_df1) + list(only_in_df2)],
            "in df2": [col in df2.columns for col in list(only_in_df1) + list(only_in_df2)],
        }
        print("column difference:\n", diff_dict)
        print("\n" * 2)

    type_matching = df1.dtypes.equals(df2.dtypes)
    if not type_matching:
        print("types do not match")
        print("df1.dtypes:\n", df1.dtypes)
        print("df2.dtypes:\n", df2.dtypes)
        print("\n" * 2)

    category_cols_df1 = df1.select_dtypes(include=["category"]).columns
    category_cols_df2 = df2.select_dtypes(include=["category"]).columns
    category_matching = category_cols_df1.equals(category_cols_df2)
    if not category_matching:
        print("category columns do not match")
        print("category_cols_df1:", category_cols_df1)
        print("category_cols_df2:", category_cols_df2)
        print("\n" * 2)

    if type_matching and column_matching:
        print("columns match, here's the difference in values")
        print("df1.compare(df2):\n", df1.compare(df2))
        print("\n" * 2)


X_seattle, y_seattle = fetch_seattle_crime_dataset()  # 0.038372s
X_seattle_offline, y_seattle_offline = fetch_seattle_crime_dataset_offline()  # 2.316111s without processing


compare_dfs(X_seattle, X_seattle_offline)  # values and types tiffer (category vs. object)
compare_dfs(y_seattle, y_seattle_offline)  # types differ (category vs. object)

import scipy

import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.impute

import statistics
import random
import math, json
import numpy as np
import pandas as pd
from joblib import dump, load
import functools
import itertools
import multiprocessing
from timeit import default_timer as timer
from pathlib import Path


RANDOM_SEED = 42


def benchmark_runtime(func):
    def wrapper(*args, **kwargs):
        t1 = timer()
        result = func(*args, **kwargs)
        t2 = timer()
        print(f"{func.__name__}() executed in {(t2-t1):.4f}s")
        return result

    return wrapper


class DataReader:
    @benchmark_runtime
    @staticmethod
    def get_congress_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "congress" / "CongressionalVotingID.shuf.lrn.csv"
        df = pd.read_csv(file_path, sep=",")
        X = df.drop("class", axis=1, inplace=False).drop("ID", axis=1, inplace=False)
        y = df["class"]
        return X, y

    @benchmark_runtime
    @staticmethod
    def get_mushroom_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "mushroom" / "secondary_data.csv"
        df = pd.read_csv(file_path, sep=";", na_values="nan", keep_default_na=False)
        X = df.drop("class", axis=1, inplace=False)
        y = df["class"]
        return X, y

    @benchmark_runtime
    @staticmethod
    def get_reviews_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "reviews" / "amazon_review_ID.shuf.lrn.csv"
        df = pd.read_csv(file_path, sep=",")
        X = df.drop("Class", axis=1, inplace=False).drop("ID", axis=1, inplace=False)
        y = df["Class"]
        return X, y

    @benchmark_runtime
    @staticmethod
    def get_seattle_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "seattle.arff"
        data = scipy.io.arff.loadarff(file_path)
        df = pd.DataFrame(data[0])

        df.drop("Report_Number", axis=1, inplace=True)  # missing from online api

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)  # bytes to str

        X = df.drop("Primary_Offense_Description", axis=1)
        y = pd.DataFrame(df["Primary_Offense_Description"])
        return X, y


def get_rand(val_prob: dict | list):
    # if list, convert to {value: probability} with uniform distribution
    if isinstance(val_prob, list):
        val_prob = {i: 1 / len(val_prob) for i in val_prob}

    total = sum(val_prob.values())
    val_prob = {k: v / total for k, v in val_prob.items()}  # normalize
    return random.choices(list(val_prob.keys()), list(val_prob.values()))[0]


def get_norm(mean, sd, nonNeg=False):
    val = random.gauss(mean, sd)
    return abs(val) if nonNeg else val


class ClassifierConfigurator:
    @staticmethod
    def get_mlp_classifier(X: pd.DataFrame) -> tuple[sklearn.neural_network.MLPClassifier, dict]:
        def get_hidden_layer_sizes(size: int) -> tuple[int, ...]:
            length = math.ceil(abs(get_norm(0, 1)))
            if length <= 1:
                length = 1
            li = [get_rand(list(e for e in range(5, size))) for _ in range(length)]
            return tuple(li)

        hidden_layer_sizes = get_hidden_layer_sizes(len(X.columns))

        kwargs = {
            "hidden_layer_sizes": hidden_layer_sizes,  # array-like of shape (n_layers - 2,), default=(100,)
            "activation": get_rand(["identity", "logistic", "tanh", "relu"]),  # {"identity", "logistic", "tanh", "relu"}, default="relu"
            "alpha": get_norm(0.0001, 0.00005, True),  # float, default=0.0001
            "batch_size": "auto",  # int, default="auto"
            "solver": get_rand(["lbfgs", "sgd", "adam"]),  # {"lbfgs", "sgd", "adam"}, default="adam"
            "learning_rate": get_rand(["constant", "invscaling", "adaptive"]),  # {"constant", "invscaling", "adaptive"}, default="constant"; only matters if solver="sgd"
            "learning_rate_init": 0.002,  # float, default=None; only when solver="sgd" or "adam"
            "power_t": get_norm(0.5, 0.2, True),  # float, default=0.5, when solver="sgd"
            "max_iter": math.ceil(get_norm(500, 100, True)),  # int, default=200
            "shuffle": get_rand({True: 0.5, False: 0.5}),  # bool, default=True
            "random_state": RANDOM_SEED,  # int, RandomState instance, default=None
            "tol": get_norm(1e-4, 5e-5, True),  # float, default=1e-4
            "verbose": False,  # bool, default=False
            "warm_start": get_rand({True: 0.3, False: 0.7}),  # bool, default=False
            "momentum": min(get_norm(0.9, 0.05, True), 0.995),  # float, default=0.9; only when solver="sgd"
            "nesterovs_momentum": get_rand({True: 0.5, False: 0.5}),  # bool, default=True; only when momentum > 0 and solver="sgd"
            "early_stopping": get_rand({True: 0.5, False: 0.5}),  # bool, default=False; only solver="sgd" or "adam"
            "validation_fraction": get_norm(0.1, 0.025, True),  # float, default=0.1
            "beta_1": min(get_norm(0.9, 0.01, True), 0.9999),  # float, default=0.9; only solver="adam"
            "beta_2": min(get_norm(0.999, 0.0001, True), 0.999999),  # float, default=0.999; only solver="adam"
            "epsilon": get_norm(1e-8, 3e-9, True),  # float, default=1e-8; only solver="adam"
            "n_iter_no_change": 10,  # int, default=10; only solver="sgd" or "adam"
            "max_fun": 15000,  # int, default=15000; only solver="lbfgs"
        }
        return sklearn.neural_network.MLPClassifier(**kwargs), kwargs

    @staticmethod
    def get_knn_classifier() -> tuple[sklearn.neighbors.KNeighborsClassifier, dict]:
        kwargs = {
            "n_neighbors": (math.ceil(get_norm(9, 4, True)) // 2) * 2 + 1,  # int, default=5
            "weights": get_rand({"uniform": 0.5, "distance": 0.5}),  # {"uniform", "distance"}, callable, or None, default="uniform"
            "algorithm": get_rand({"auto": 0.40, "ball_tree": 0.25, "kd_tree": 0.25, "brute": 0.10}),  # {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
            "leaf_size": math.ceil(get_norm(30, 10, True)),  # int, default=30, for kd and ball
            "p": get_rand({1: 0.30, 2: 0.30, 1.5: 0.2, 4: 0.2}),  # float, default=2
            "metric": "minkowski",  # str or callable, default="minkowski"
            "metric_params": None,  # dict, default=None
            "n_jobs": 10,  # int, default=None
        }
        return sklearn.neighbors.KNeighborsClassifier(**kwargs), kwargs

    @staticmethod
    def get_rf_classifier() -> tuple[sklearn.ensemble.RandomForestClassifier, dict]:
        kwargs = {
            "n_estimators": math.ceil(get_norm(15, 7, True)),  # int, default=100
            "criterion": get_rand(["gini", "entropy", "log_loss"]),  # "gini", "entropy", "log_loss" defaukt=gini
            "max_depth": math.ceil(get_norm(12, 4, True)),  # int, default=None CAUTION: default value None is dangerous for the pc
            "min_samples_split": get_rand({2: 0.50, 4: 0.20, 6: 0.20, 8: 0.10}),  # int or float, default=2
            "min_samples_leaf": get_rand({1: 0.50, 2: 0.20, 3: 0.20, 5: 0.10}),  # int or float, default=1
            "min_weight_fraction_leaf": 0.0,  # float, default=0.0
            "max_features": get_rand(["sqrt", "log2", None]),  # "sqrt", "log2", None, int, or float, default="sqrt"
            "max_leaf_nodes": None,  # int, default=None
            "min_impurity_decrease": 0.0,  # float, default=0.0
            "bootstrap": get_rand({True: 0.5, False: 0.5}),  # bool, default=True
            "oob_score": False,  # bool or callable, default=False
            "n_jobs": 10,  # int, default=None
            "random_state": RANDOM_SEED,  # int, RandomState instance, or None, default=None
            "verbose": 0,  # int, default=0
            "warm_start": get_rand({True: 0.3, False: 0.7}),  # bool, default=False
            "class_weight": get_rand({None: 1, "balanced": 0, "balanced_subsample": 0}),  # {"balanced", "balanced_subsample"}, dict or list of dicts, default=None
            "ccp_alpha": get_rand({0: 0.4, 0.05: 0.25, 0.5: 0.25, 2: 0.1}),  # non-negative float, default=0.0
            "max_samples": None,  # int or float, default=None
            "monotonic_cst": None,  # array-like of int of shape (n_features), default=None
        }
        return sklearn.ensemble.RandomForestClassifier(**kwargs), kwargs


def encode_one_hot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # use OneHotEncoder instead of get_dummies
    one_hot = sklearn.preprocessing.OneHotEncoder()
    encoded = one_hot.fit_transform(df[[column]])
    one_hot_df = pd.DataFrame(encoded.toarray(), columns=one_hot.get_feature_names_out([column]))
    return pd.concat([df, one_hot_df], axis=1).drop(column, axis=1, inplace=False)


def map_to_dict(X: pd.DataFrame, col: str, label_map: dict) -> None:
    X[col] = X[col].map(label_map)


def map_to_int(X: pd.DataFrame | pd.Series, col: str) -> None:
    X[col] = sklearn.preprocessing.LabelEncoder().fit_transform(X[col])  # map values to integers


def scale_mean(X: pd.DataFrame, col: str) -> None:
    X[col] = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(X[[col]])


def scale_minmax(X: pd.DataFrame, col: str) -> None:
    X[col] = sklearn.preprocessing.MinMaxScaler().fit_transform(X[[col]])


def impute_mode(X: pd.DataFrame, col: str, missing_val: str) -> None:
    X.loc[X[col] == missing_val, col] = statistics.mode(X[X[col] != missing_val][col])


class Preprocessor:
    @staticmethod
    def fix_congress_dataset(X: pd.DataFrame, y: pd.Series, imputing: bool, labeling: bool, scaling: str | None) -> tuple[pd.DataFrame, pd.Series]:
        to_impute = X.columns
        for col in to_impute:
            if imputing:
                impute_mode(X, col, "unknown")
            if labeling:
                map_to_dict(X, col, {"y": 1, "n": 0, "unknown": 2})
            else:
                X = encode_one_hot(X, col)  # might not make a lot of sense for this dataset

        y = pd.Series(np.array(sklearn.preprocessing.LabelEncoder().fit_transform(y)))
        return X, y

    @staticmethod
    def fix_mushroom_dataset(X: pd.DataFrame, y: pd.Series, imputing: bool, labeling: bool, scaling: str | None) -> tuple[pd.DataFrame, pd.Series]:
        to_impute = [
            "cap-shape",
            "cap-surface",
            "cap-color",
            "gill-attachment",
            "gill-spacing",
            "gill-color",
            "stem-root",
            "stem-color",
            "stem-surface",
            "veil-type",
            "does-bruise-or-bleed",
            "veil-color",
            "has-ring",
            "ring-type",
            "spore-print-color",
            "habitat",
            "season",
        ]
        for col in to_impute:
            if imputing:
                impute_mode(X, col, "")
            if labeling:
                map_to_int(X, col)
            else:
                X = encode_one_hot(X, col)

        to_scale = ["cap-diameter", "stem-height", "stem-width"]
        for i in to_scale:
            if scaling == "mean":
                scale_mean(X, i)
            elif scaling == "minmax":
                scale_minmax(X, i)

        y = pd.Series(np.array(sklearn.preprocessing.LabelEncoder().fit_transform(y)))
        return X, y

    @staticmethod
    def fix_reviews_dataset(X: pd.DataFrame, y: pd.Series, imputing: bool, labeling: bool, scaling: str | None) -> tuple[pd.DataFrame, pd.Series]:
        to_scale = X.columns
        for i in to_scale:
            if scaling == "mean":
                scale_mean(X, i)
            elif scaling == "minmax":
                scale_minmax(X, i)

        y = pd.Series(np.array(sklearn.preprocessing.LabelEncoder().fit_transform(y)))
        return X, y

    @staticmethod
    def fix_seattle_dataset(X: pd.DataFrame, y: pd.Series, imputing: bool, labeling: bool, scaling: str | None) -> tuple[pd.DataFrame, pd.Series]:
        # regardless of whether mean or minimax is set, minimax is used
        if scaling is not None:
            X["Occurred_Time"] /= 2400
            X["Reported_Time"] /= 2400

        missing_values = ["?", "UNKNOWN", "?", "?", "UNKNOWN"]
        cols = ["Crime_Subcategory", "Precinct", "Sector", "Beat", "Neighborhood"]
        for missing_val, col in zip(missing_values, cols):
            if imputing:
                impute_mode(X, col, missing_val)
            if labeling:
                map_to_int(X, col)
            else:
                X = encode_one_hot(X, col)

        map_to_int(y, "Primary_Offense_Description")
        y = pd.Series(y["Primary_Offense_Description"])

        # missing Values
        imputer = sklearn.impute.SimpleImputer(strategy="mean")
        imputer.fit(X)
        imputed_df = pd.DataFrame(imputer.transform(X), columns=X.columns)
        X["Occurred_Time"] = imputed_df["Occurred_Time"]
        X["Reported_Time"] = imputed_df["Reported_Time"]
        return X, y


# we can't share this among processes, because it's not picklable
X_congress, y_congress = DataReader.get_congress_dataset()
X_mushroom, y_mushroom = DataReader.get_mushroom_dataset()
X_reviews, y_reviews = DataReader.get_reviews_dataset()
X_seattle, y_seattle = DataReader.get_seattle_dataset()


def run(arg):
    print(f"running... {arg=}")
    dataset, scaling, labeling, imputing, classifier_type = arg

    # 1) pick dataset, preprocess data
    X, y = None, None
    if dataset == "congress":
        X, y = X_congress.copy(), y_congress.copy()
        X, y = Preprocessor.fix_congress_dataset(X, y, imputing, labeling, scaling)
    elif dataset == "mushroom":
        X, y = X_mushroom.copy(), y_mushroom.copy()
        X, y = Preprocessor.fix_mushroom_dataset(X, y, imputing, labeling, scaling)
    elif dataset == "reviews":
        X, y = X_reviews.copy(), y_reviews.copy()
        X, y = Preprocessor.fix_reviews_dataset(X, y, imputing, labeling, scaling)
    elif dataset == "seattle":
        X, y = X_seattle.copy(), y_seattle.copy()
        X, y = Preprocessor.fix_seattle_dataset(X, y, imputing, labeling, scaling)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    assert X is not None and y is not None
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)

    # 2) pick model
    classifier = None
    config = None
    if classifier_type == "mlp":
        classifier, config = ClassifierConfigurator.get_mlp_classifier(X)
    elif classifier_type == "knn":
        classifier, config = ClassifierConfigurator.get_knn_classifier()
    elif classifier_type == "rf":
        classifier, config = ClassifierConfigurator.get_rf_classifier()
    else:
        raise ValueError(f"unknown classifier: {classifier_type}")
    assert classifier is not None and config is not None
    assert isinstance(classifier, sklearn.base.ClassifierMixin)

    # 3) split data, train model, evaluate model (time, accuracy) -> need more metrics
    time = None
    acc = None
    try:
        start_time = timer()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        end_time = timer()
        time = end_time - start_time

        if MULTICORE:
            # TODO implement multicore safe file writing
            print(
                f"dataset: {dataset}, scaling: {scaling}, labeling: {labeling}, imputing: {imputing}, cross_val: {cross_val}, time: {time}, acc: {acc}, classifier_type: {classifier_type}, classifier_config: {config}"
            )
            pass
        else:
            with open(f"reports\\AUTO\\{classifier_type}_{dataset}.json", "r") as f:
                contents = f.readlines()

            def add_quotes(val):
                return '"' + val + '"'

            if len(contents) > 2:
                contents[-2] = contents[-2].replace("\n", ",\n")

            contents.insert(
                -1,
                f'\t{"{"}"id": "", "Settings": {json.dumps(arg)}, "Scaling": {add_quotes(scaling) if scaling else scaling}, "Labeling": "{labeling}", "Impute": "{imputing}", "Holdout": {"{"}"Duration": {time}, "Accuracy": {acc}{"}"}{"}"}\r',
            )

            with open(f"reports\\AUTO\\{classifier_type}_{dataset}.json", "w") as f:
                contents = "".join(contents)
                f.write(contents)
    except Exception as e:
        print("error:", e)


if __name__ == "__main__":
    # anything within this function will be only ran by the master process

    COMBINATIONS = {
        "dataset": ["congress", "mushroom", "reviews", "seattle"],
        "scaling": ["mean", "minmax", None],
        "labeling": [True, False],
        "imputing": [True, False],
        "classifier_type": ["mlp", "knn", "rf"],
    }
    random_combinations = list(itertools.product(*COMBINATIONS.values()))
    random.shuffle(random_combinations)
    print(f"{len(random_combinations)=}")

    MULTICORE = False
    ITERS = 10

    if MULTICORE:
        with multiprocessing.Pool() as pool:
            pool.map(run, random_combinations[:ITERS])

    if not MULTICORE:
        while True:
            run(random.choice(random_combinations))

import scipy

import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.impute

import traceback
import statistics
import random
import math
import json
import numpy as np
import pandas as pd
import functools
import itertools
import hashlib
import multiprocessing
from timeit import default_timer as timer
from pathlib import Path
import signal
from contextlib import contextmanager


RANDOM_SEED = 42
MULTICORE = False


def benchmark_runtime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = timer()
        result = func(*args, **kwargs)
        t2 = timer()
        print(f"{func.__name__}() executed in {(t2-t1):.4f}s")
        return result

    return wrapper


class TimeoutException(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            @contextmanager
            def time_limit(seconds):
                def signal_handler(signum, frame):
                    raise TimeoutException("Function timed out")

                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)

            with time_limit(seconds):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class DataReader:
    @staticmethod
    def get_congress_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "congress" / "CongressionalVotingID.shuf.lrn.csv"
        df = pd.read_csv(file_path, sep=",")
        X = df.drop("class", axis=1, inplace=False).drop("ID", axis=1, inplace=False)
        y = df["class"]
        return X, y

    @staticmethod
    def get_mushroom_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "mushroom" / "secondary_data.csv"
        df = pd.read_csv(file_path, sep=";", na_values="nan", keep_default_na=False)
        X = df.drop("class", axis=1, inplace=False)
        y = df["class"]
        return X, y

    @staticmethod
    def get_reviews_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "reviews" / "amazon_review_ID.shuf.lrn.csv"
        df = pd.read_csv(file_path, sep=",")
        X = df.drop("Class", axis=1, inplace=False).drop("ID", axis=1, inplace=False)
        y = df["Class"]
        return X, y

    @staticmethod
    def get_seattle_dataset() -> tuple[pd.DataFrame, pd.Series]:
        file_path = Path(__file__).resolve().parent / "data" / "seattle.arff"
        data = scipy.io.arff.loadarff(file_path)
        df = pd.DataFrame(data[0])

        df.drop("Report_Number", axis=1, inplace=True)  # missing from online api

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)  # bytes to str

        X = df.drop("Primary_Offense_Description", axis=1, inplace=False)
        y = df["Primary_Offense_Description"]
        return X, y


class CConfig:
    @staticmethod
    def get_rand(val_prob: dict | list):
        # if list, convert to {value: probability} with uniform distribution
        if isinstance(val_prob, list):
            val_prob = {i: 1 / len(val_prob) for i in val_prob}

        total = sum(val_prob.values())
        val_prob = {k: v / total for k, v in val_prob.items()}  # normalize
        return random.choices(list(val_prob.keys()), list(val_prob.values()))[0]

    @staticmethod
    def get_norm(mean, sd, nonNeg=False):
        val = random.gauss(mean, sd)
        return abs(val) if nonNeg else val

    @staticmethod
    def get_mlp_classifier(X: pd.DataFrame) -> tuple[sklearn.neural_network.MLPClassifier, dict]:
        def get_hidden_layer_sizes(size: int) -> tuple[int, ...]:
            length = math.ceil(abs(CConfig.get_norm(0, 1)))
            if length <= 1:
                length = 1
            li = [CConfig.get_rand(list(e for e in range(5, size))) for _ in range(length)]
            return tuple(li)

        hidden_layer_sizes = get_hidden_layer_sizes(len(X.columns))

        kwargs = {
            "hidden_layer_sizes": hidden_layer_sizes,  # array-like of shape (n_layers - 2,), default=(100,)
            "activation": CConfig.get_rand(["identity", "logistic", "tanh", "relu"]),  # {"identity", "logistic", "tanh", "relu"}, default="relu"
            "alpha": CConfig.get_norm(0.0001, 0.00005, True),  # float, default=0.0001
            "batch_size": "auto",  # int, default="auto"
            "solver": CConfig.get_rand(["lbfgs", "sgd", "adam"]),  # {"lbfgs", "sgd", "adam"}, default="adam"
            "learning_rate": CConfig.get_rand(["constant", "invscaling", "adaptive"]),  # {"constant", "invscaling", "adaptive"}, default="constant"; only matters if solver="sgd"
            "learning_rate_init": 0.002,  # float, default=None; only when solver="sgd" or "adam"
            "power_t": CConfig.get_norm(0.5, 0.2, True),  # float, default=0.5, when solver="sgd"
            "max_iter": math.ceil(CConfig.get_norm(500, 100, True)),  # int, default=200
            "shuffle": CConfig.get_rand({True: 0.5, False: 0.5}),  # bool, default=True
            "random_state": RANDOM_SEED,  # int, RandomState instance, default=None
            "tol": CConfig.get_norm(1e-4, 5e-5, True),  # float, default=1e-4
            "verbose": False,  # bool, default=False
            "warm_start": CConfig.get_rand({True: 0.3, False: 0.7}),  # bool, default=False
            "momentum": min(CConfig.get_norm(0.9, 0.05, True), 0.995),  # float, default=0.9; only when solver="sgd"
            "nesterovs_momentum": CConfig.get_rand({True: 0.5, False: 0.5}),  # bool, default=True; only when momentum > 0 and solver="sgd"
            "early_stopping": CConfig.get_rand({True: 0.5, False: 0.5}),  # bool, default=False; only solver="sgd" or "adam"
            "validation_fraction": CConfig.get_norm(0.1, 0.025, True),  # float, default=0.1
            "beta_1": min(CConfig.get_norm(0.9, 0.01, True), 0.9999),  # float, default=0.9; only solver="adam"
            "beta_2": min(CConfig.get_norm(0.999, 0.0001, True), 0.999999),  # float, default=0.999; only solver="adam"
            "epsilon": CConfig.get_norm(1e-8, 3e-9, True),  # float, default=1e-8; only solver="adam"
            "n_iter_no_change": 10,  # int, default=10; only solver="sgd" or "adam"
            "max_fun": 15000,  # int, default=15000; only solver="lbfgs"
        }
        return sklearn.neural_network.MLPClassifier(**kwargs), kwargs

    @staticmethod
    def get_knn_classifier() -> tuple[sklearn.neighbors.KNeighborsClassifier, dict]:
        kwargs = {
            "n_neighbors": (math.ceil(CConfig.get_norm(9, 4, True)) // 2) * 2 + 1,  # int, default=5
            "weights": CConfig.get_rand({"uniform": 0.5, "distance": 0.5}),  # {"uniform", "distance"}, callable, or None, default="uniform"
            "algorithm": CConfig.get_rand({"auto": 0.40, "ball_tree": 0.25, "kd_tree": 0.25, "brute": 0.10}),  # {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
            "leaf_size": math.ceil(CConfig.get_norm(30, 10, True)),  # int, default=30, for kd and ball
            "p": CConfig.get_rand({1: 0.30, 2: 0.30, 1.5: 0.2, 4: 0.2}),  # float, default=2
            "metric": "minkowski",  # str or callable, default="minkowski"
            "metric_params": None,  # dict, default=None
            "n_jobs": None if MULTICORE else multiprocessing.cpu_count(),  # int, default=None
        }
        return sklearn.neighbors.KNeighborsClassifier(**kwargs), kwargs

    @staticmethod
    def get_rf_classifier() -> tuple[sklearn.ensemble.RandomForestClassifier, dict]:
        kwargs = {
            "n_estimators": math.ceil(CConfig.get_norm(15, 7, True)),  # int, default=100
            "criterion": CConfig.get_rand(["gini", "entropy", "log_loss"]),  # "gini", "entropy", "log_loss" defaukt=gini
            "max_depth": math.ceil(CConfig.get_norm(12, 4, True)),  # int, default=None CAUTION: default value None is dangerous for the pc
            "min_samples_split": CConfig.get_rand({2: 0.50, 4: 0.20, 6: 0.20, 8: 0.10}),  # int or float, default=2
            "min_samples_leaf": CConfig.get_rand({1: 0.50, 2: 0.20, 3: 0.20, 5: 0.10}),  # int or float, default=1
            "min_weight_fraction_leaf": 0.0,  # float, default=0.0
            "max_features": CConfig.get_rand(["sqrt", "log2", None]),  # "sqrt", "log2", None, int, or float, default="sqrt"
            "max_leaf_nodes": None,  # int, default=None
            "min_impurity_decrease": 0.0,  # float, default=0.0
            "bootstrap": CConfig.get_rand({True: 0.5, False: 0.5}),  # bool, default=True
            "oob_score": False,  # bool or callable, default=False
            "n_jobs": None if MULTICORE else multiprocessing.cpu_count(),  # int, default=None
            "random_state": RANDOM_SEED,  # int, RandomState instance, or None, default=None
            "verbose": 0,  # int, default=0
            "warm_start": CConfig.get_rand({True: 0.3, False: 0.7}),  # bool, default=False
            "class_weight": CConfig.get_rand({None: 1, "balanced": 0, "balanced_subsample": 0}),  # {"balanced", "balanced_subsample"}, dict or list of dicts, default=None
            "ccp_alpha": CConfig.get_rand({0: 0.4, 0.05: 0.25, 0.5: 0.25, 2: 0.1}),  # non-negative float, default=0.0
            "max_samples": None,  # int or float, default=None
            "monotonic_cst": None,  # array-like of int of shape (n_features), default=None
        }
        return sklearn.ensemble.RandomForestClassifier(**kwargs), kwargs


class Preprocessor:
    @staticmethod
    def encode_one_hot(df: pd.DataFrame, column: str) -> pd.DataFrame:
        # use OneHotEncoder instead of get_dummies
        one_hot = sklearn.preprocessing.OneHotEncoder()
        encoded: scipy.sparse.csr.csr_matrix = one_hot.fit_transform(df[[column]])
        one_hot_df = pd.DataFrame(encoded.toarray(), columns=one_hot.get_feature_names_out([column]))
        return pd.concat([df, one_hot_df], axis=1).drop(column, axis=1, inplace=False)

    @staticmethod
    def map_to_dict(X: pd.DataFrame, col: str, label_map: dict) -> None:
        X[col] = X[col].map(label_map)

    @staticmethod
    def map_to_int(X: pd.DataFrame | pd.Series, col: str) -> None:
        X[col] = sklearn.preprocessing.LabelEncoder().fit_transform(X[col])  # map values to integers

    @staticmethod
    def scale_mean(X: pd.DataFrame, col: str) -> None:
        X[col] = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(X[[col]])

    @staticmethod
    def scale_minmax(X: pd.DataFrame, col: str) -> None:
        X[col] = sklearn.preprocessing.MinMaxScaler().fit_transform(X[[col]])

    @staticmethod
    def impute_mode(X: pd.DataFrame, col: str, missing_val: str) -> None:
        X.loc[X[col] == missing_val, col] = statistics.mode(X[X[col] != missing_val][col])

    @staticmethod
    def fix_congress_dataset(X: pd.DataFrame, y: pd.Series, imputing: bool, labeling: bool, scaling: str | None) -> tuple[pd.DataFrame, pd.Series]:
        to_impute = X.columns
        for col in to_impute:
            if imputing:
                Preprocessor.impute_mode(X, col, "unknown")
            if labeling:
                Preprocessor.map_to_dict(X, col, {"y": 1, "n": 0, "unknown": 2})
            else:
                X = Preprocessor.encode_one_hot(X, col)  # might not make a lot of sense for this dataset

        y = pd.Series(np.array(sklearn.preprocessing.LabelEncoder().fit_transform(y)))  # type: ignore
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
                Preprocessor.impute_mode(X, col, "")
            if labeling:
                Preprocessor.map_to_int(X, col)
            else:
                X = Preprocessor.encode_one_hot(X, col)

        to_scale = ["cap-diameter", "stem-height", "stem-width"]
        for i in to_scale:
            if scaling == "mean":
                Preprocessor.scale_mean(X, i)
            elif scaling == "minmax":
                Preprocessor.scale_minmax(X, i)

        y = pd.Series(np.array(sklearn.preprocessing.LabelEncoder().fit_transform(y)))
        return X, y

    @staticmethod
    def fix_reviews_dataset(X: pd.DataFrame, y: pd.Series, imputing: bool, labeling: bool, scaling: str | None) -> tuple[pd.DataFrame, pd.Series]:
        to_scale = X.columns
        for i in to_scale:
            if scaling == "mean":
                Preprocessor.scale_mean(X, i)
            elif scaling == "minmax":
                Preprocessor.scale_minmax(X, i)

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
                Preprocessor.impute_mode(X, col, missing_val)
            if labeling:
                Preprocessor.map_to_int(X, col)
            else:
                X = Preprocessor.encode_one_hot(X, col)

        # map y to int
        y = pd.Series(np.array(sklearn.preprocessing.LabelEncoder().fit_transform(y)))
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)

        # missing Values
        imputer = sklearn.impute.SimpleImputer(strategy="mean")
        imputer.fit(X)
        imputed_df = pd.DataFrame(imputer.transform(X), columns=X.columns)  # type: ignore
        X["Occurred_Time"] = imputed_df["Occurred_Time"]
        X["Reported_Time"] = imputed_df["Reported_Time"]
        return X, y


# we can't share this among processes, because it's not pickleable
X_congress, y_congress = DataReader.get_congress_dataset()
X_mushroom, y_mushroom = DataReader.get_mushroom_dataset()
X_reviews, y_reviews = DataReader.get_reviews_dataset()
X_seattle, y_seattle = DataReader.get_seattle_dataset()


# @timeout(60 if MULTICORE else 300)
def run_worker(arg) -> dict:
    print(f"running... {arg=}")
    dataset, scaling, labeling, imputing, classifier_type, cross_val = arg

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
        X, y = Preprocessor.fix_seattle_dataset(X, y, imputing, labeling, scaling)  # type: ignore
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    assert X is not None and y is not None
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
    assert not np.isnan(y).any() and not np.isinf(y).any()
    assert not np.isnan(X).any().any() and not np.isinf(X).any().any()
    assert X.shape[0] == y.shape[0]
    print("preprocessed data")

    # 2) pick model
    classifier = None
    config = None
    if classifier_type == "mlp":
        classifier, config = CConfig.get_mlp_classifier(X)
    elif classifier_type == "knn":
        classifier, config = CConfig.get_knn_classifier()
    elif classifier_type == "rf":
        classifier, config = CConfig.get_rf_classifier()
    else:
        raise ValueError(f"unknown classifier: {classifier_type}")
    assert classifier is not None and config is not None
    assert isinstance(classifier, sklearn.base.ClassifierMixin)
    print("selected model")

    # 3) split data, train model, evaluate model
    output = {}
    output["args"] = {
        "dataset": dataset,
        "scaling": scaling,
        "labeling": labeling,
        "imputing": imputing,
        "classifier_type": classifier_type,
        "cross_val": cross_val,
    }
    output["classifier_config"] = config
    if cross_val:
        NUM_SPLITS = 3
        start_time = timer()
        k_fold = sklearn.model_selection.KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        print("split data")
        y_pred = sklearn.model_selection.cross_val_predict(classifier, X, y, cv=k_fold, n_jobs=None if MULTICORE else multiprocessing.cpu_count())
        print("ran predict")
        end_time = timer()
        output["validation"] = f"{NUM_SPLITS}-fold cross validation"
        output["time"] = end_time - start_time

        output["accuracy_score"] = sklearn.metrics.accuracy_score(y, y_pred, normalize=True)
        output["balanced_accuracy_score"] = sklearn.metrics.balanced_accuracy_score(y, y_pred, adjusted=True)
        output["precision_score"] = sklearn.metrics.precision_score(y, y_pred, average="weighted", zero_division=0, labels=np.unique(y_pred))
        # output["average_precision_score"] = sklearn.metrics.average_precision_score(y, y_pred, average="micro")  # <-- breaks for no reason
        output["recall_score"] = sklearn.metrics.recall_score(y, y_pred, average="weighted", zero_division=0, labels=np.unique(y_pred))
        output["zero-one-loss"] = sklearn.metrics.zero_one_loss(y, y_pred, normalize=True)

    else:
        SPLIT_RATIO = 0.2
        start_time = timer()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=SPLIT_RATIO, random_state=RANDOM_SEED, shuffle=True)
        print("split data")
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        print("ran predict")
        end_time = timer()
        output["validation"] = f"houldout validation, {(1-SPLIT_RATIO)*100}% train, {SPLIT_RATIO*100}% test"
        output["time"] = end_time - start_time

        output["accuracy_score"] = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True)
        output["balanced_accuracy_score"] = sklearn.metrics.balanced_accuracy_score(y_test, y_pred, adjusted=True)
        output["precision_score"] = sklearn.metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0, labels=np.unique(y_pred))
        output["recall_score"] = sklearn.metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0, labels=np.unique(y_pred))
        output["zero-one-loss"] = sklearn.metrics.zero_one_loss(y_test, y_pred, normalize=True)

    # 4) save results
    def truncate_floats(obj, decimals=10):
        if isinstance(obj, float):
            return round(obj, decimals)
        elif isinstance(obj, dict):
            return {k: truncate_floats(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [truncate_floats(e, decimals) for e in obj]
        else:
            return str(obj)

    def save_results(output: dict) -> None:
        output_path = Path(__file__).resolve().parent / "output" / f"results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)

        if output_path.stat().st_size == 0:
            with open(output_path, "w") as f:
                f.write('{"results": [\n')
        with open(output_path, "a") as f:
            json.dump(output, f)
            f.write(",\n")

    output = truncate_floats(output)
    assert isinstance(output, dict)
    print(f"\033[92m{output['time']:.4f}s:\n{output}\033[0m")
    save_results(output)
    return output


def run_master():
    def get_progress() -> set:
        progress_path = Path(__file__).resolve().parent / "output" / f"results-merged.json"
        assert progress_path.exists()
        progress: dict = json.loads(progress_path.read_text())
        progress = [r["args"] for r in progress["results"]]  # type: ignore
        progress = [list(r.values()) for r in progress]  # type: ignore
        progress = [[True if e == "True" else False if e == "False" else e for e in r] for r in progress]  # type: ignore
        progress = [[None if e == "None" else e for e in r] for r in progress]  # type: ignore
        progress = [tuple(r) for r in progress]  # type: ignore
        return set(progress)

    COMBINATIONS = {
        "dataset": ["congress", "mushroom", "reviews", "seattle"],
        "scaling": ["mean", "minmax", None],
        "labeling": [True, False],
        "imputing": [True, False],
        "classifier_type": ["mlp", "knn", "rf"],
        "cross_val": [True, False],
    }
    random_combinations = list(itertools.product(*COMBINATIONS.values()))
    random.shuffle(random_combinations)
    print(f"total: {len(random_combinations)}")
    random_combinations = [r for r in random_combinations if r not in get_progress()]
    print(f"remaining: {len(random_combinations)}")

    ITERS = len(random_combinations) - 0
    if MULTICORE:
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cores) as pool:
            pool.map(run_worker, random_combinations[:ITERS])
    else:
        for arg in random_combinations[:ITERS]:
            try:
                run_worker(arg)
            except TimeoutException:
                print("\033[91mtimeout\033[0m")
            except Exception as e:
                print(f"\033[91merror: {e}\033[0m")
                traceback.print_exc()


if __name__ == "__main__":
    run_master()

from pathlib import Path
from joblib import dump

import click
import pandas as pd
import numpy as np

import mlflow.sklearn
import mlflow

from sklearn.model_selection import cross_validate
from .data import get_dataset
from .pipeline import create_pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


import sys
import os
import warnings


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--feature_select_method", default=0, type=int, show_default=True)
@click.option("--random-state", default=42, type=int, show_default=True)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--k_folds",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--second-model",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--n_estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    type=str,
    show_default=True,
)
@click.option(
    "--min_samples_leaf",
    default=0.1,
    type=float,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
def train_nested_cv(
    dataset_path: Path,
    save_model_path: Path,
    feature_select_method: int,
    random_state: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    k_folds: int,
    second_model: bool,
    n_estimators: int,
    criterion: str,
    min_samples_leaf: float,
    max_depth: int,
) -> None:

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

        features, target = get_dataset(dataset_path, feature_select_method)

    if second_model:
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )

        space_rf = {
            "n_estimators": [10, 100, 500],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [0.1, 0.2, 0.3, 0.4],
            "max_depth": [5, 6, 7, 8, 9, 10],
        }
        search_rf = GridSearchCV(
            rf, space_rf, scoring="accuracy", n_jobs=1, cv=cv_inner, refit=True
        )

        roc_auc_score_rf, accuracy_score_rf, f1_score_rf = list(), list(), list()

        for train_index, test_index in cv_outer.split(features, target):
            X_train, X_test = (
                features.iloc[train_index, :],
                features.iloc[test_index, :],
            )
            y_train, y_test = target[train_index], target[test_index]
        search_rf.fit(X_train, y_train)
        best_rf = search_rf.best_estimator_
        accuracy_rf = accuracy_score(best_rf.predict(X_test), y_test)
        roc_rf = roc_auc_score(
            y_test,
            best_rf.predict_proba(X_test),
            multi_class="ovr",
            average="macro",
        )
        f1_rf = f1_score(y_test, best_rf.predict(X_test), average="macro")
        roc_auc_score_rf.append(roc_rf)
        accuracy_score_rf.append(accuracy_rf)
        f1_score_rf.append(f1_rf)
        print("ROC AUC score: ", np.mean(roc_auc_score_rf))
        print("Accuracy score: ", np.mean(accuracy_score_rf))
        print("F1 score: ", np.mean(f1_score_rf))

    else:
        space_logreg = {
            "max_iter": [50, 100, 150, 200],
            "C": [0.01, 0.1, 0.2, 0.3, 0.5, 2],
        }
        cv_inner_2 = KFold(n_splits=3, shuffle=True, random_state=random_state)
        logreg = LogisticRegression(
            max_iter=max_iter,
            C=logreg_c,
        )

        search_logreg = GridSearchCV(
            logreg,
            space_logreg,
            scoring="accuracy",
            n_jobs=1,
            cv=cv_inner_2,
            refit=True,
        )
        cv_outer_2 = KFold(n_splits=10, shuffle=True, random_state=random_state)

        roc_auc_score_logreg, accuracy_score_logreg, f1_score_logreg = (
            list(),
            list(),
            list(),
        )

        for train_index, test_index in cv_outer_2.split(features, target):
            X_train, X_test = (
                features.iloc[train_index, :],
                features.iloc[test_index, :],
            )
            y_train, y_test = target[train_index], target[test_index]
        search_logreg.fit(X_train, y_train)
        best_logreg = search_logreg.best_estimator_
        accuracy_logreg = accuracy_score(best_logreg.predict(X_test), y_test)
        roc_logreg = roc_auc_score(
            y_test,
            best_logreg.predict_proba(X_test),
            multi_class="ovr",
            average="macro",
        )
        f1_logreg = f1_score(y_test, best_logreg.predict(X_test), average="macro")
        roc_auc_score_logreg.append(roc_logreg)
        accuracy_score_logreg.append(accuracy_logreg)
        f1_score_logreg.append(f1_logreg)

        print("ROC AUC score: ", np.mean(roc_auc_score_logreg))
        print("Accuracy score: ", np.mean(accuracy_score_logreg))
        print("F1 score: ", np.mean(f1_score_logreg))

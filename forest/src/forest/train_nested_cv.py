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
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
        scores_rf = cross_validate(
            search_rf,
            features,
            target,
            scoring=("accuracy", "f1_macro", "roc_auc_ovr"),
            error_score="raise",
            cv=cv_outer,
            n_jobs=-1,
        )

        print(
            "Score Accuracy RandomForestClassifier: %.3f (%.3f)"
            % (np.mean(scores_rf), np.std(scores_rf))
        )

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
        scores_logreg = cross_validate(
            search_logreg,
            features,
            target,
            scoring=("accuracy", "f1_macro", "roc_auc_ovr"),
            error_score="raise",
            cv=cv_outer_2,
            n_jobs=-1,
        )

        print(
            "Score LogisticRegression: %.3f (%.3f)"
            % (np.mean(scores_logreg), np.std(scores_logreg))
        )

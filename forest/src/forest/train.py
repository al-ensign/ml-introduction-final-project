from pathlib import Path
from joblib import dump

import click
import pandas as pd
import pandas_profiling

import mlflow.sklearn
import mlflow

from sklearn.model_selection import cross_validate
from .data import get_dataset
from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True
)

@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True
)

@click.option(
    "--feature-select_method",
    default=0,
    type=int,
    show_default=True
)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True
)

@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)

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
    default=1.0,
    type=float,
    show_default=True,
)

@click.option(
    "--max-depth",
    default=5,
    type=int,
    show_default=True,
)

def train(
    dataset_path: Path,
    save_model_path: Path,
    feature_select: int,
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

    features, target = get_dataset(dataset_path, feature_select)

    with mlflow.start_run():
    
        pipeline = create_pipeline(
            use_scaler, max_iter, logreg_c, random_state, second_model, n_estimators, criterion, min_samples_leaf, max_depth
            )

            cv_results = cross_validate(pipeline, features, target, cv=k_folds, scoring=('accuracy', 'f1', 'roc_auc'),)
        
        if second_model:
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("model_type", "RandomForestClassifier")

        else:
            mlflow.log_param("use_scaler", use_scaler)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
            mlflow.log_param("model_type", "LogisticRegression")
             
        mlflow.log_param("feature_select_method", feature_select_method)
        mlflow.log_param("k_folds", k_folds)
            
        mlflow.log_metric("accuracy", cv_results['test_accuracy'].mean())
        mlflow.log_metric("f1_score", cv_results['test_f1'].mean())
        mlflow.log_metric("roc_value", cv_results['test_roc_auc'].mean())
    
        dump(pipeline, save_model_path)
        click.echo(f"Saved to {save_model_path}.")
        click.echo(f"Cross-validation scores :") 
        click.echo(f"accuracy : {cv_results['test_accuracy'].mean()}.") 
        click.echo(f"f1_score : {cv_results['test_f1'].mean()}.")
        click.echo(f"roc_value : {cv_results['test_roc_auc'].mean()}.") 
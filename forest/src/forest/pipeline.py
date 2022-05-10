from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def create_pipeline(
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    random_state: int,
    second_model: bool,
    n_estimators: int,
    criterion: str,
    min_samples_leaf: float,
    max_depth: int,
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(("scaler", MinMaxScaler()))

    if second_model:
        pipeline_steps.append(
            (
                "randomforest",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            )
        )
    else:
        pipeline_steps.append(
            (
                "logreg",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=max_iter,
                    C=logreg_c,
                ),
            )
        )

    return Pipeline(steps=pipeline_steps)

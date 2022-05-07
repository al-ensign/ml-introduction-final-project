from pathlib import Path
from joblib import dump

import click
import pandas as pd
import pandas_profiling

from .data import get_data


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True
)

def profile_report(
    dataset_path: Path,
    ) -> None:

    data = get_data(dataset_path)
    profile = data.profile_report(title="Forest Cover Type Prediction - Pandas Profiling Report")
    profile.to_file("Forest_report.html")
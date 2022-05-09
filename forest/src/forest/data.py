import logging
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


#Remove test_split devision from demo_project code 
#Will use K-Fold cross-validation in train
def get_dataset(
    csv_path: Path, 
    feature_select_method: int,
    ) -> Tuple[pd.DataFrame, pd.Series]: 

    dataset = pd.read_csv(csv_path)

    features = data.drop(columns=["Cover_Type"])    
    target = data['Cover_Type']

    if feature_select_method == 0:
        return features, target

    elif feature_select_method == 1:

        extra_trees = ExtraTreesClassifier(n_estimators=40)
        extra_trees.fit(features, target)
        model = SelectFromModel(extra_trees, prefit=True)

    elif feature_select_method == 2:

        model = BorutaPy(RandomForestClassifier(max_depth=5), 
                                n_estimators=200, 
                                min_samples_leaf=0.2,
                                criterion="gini"
                                max_iter=100,
                                random_state=42)

        model.fit(np.asarray(features), np.asarray(target))

    return model.transform(np.asarray(features)), target

    return features, target

#Get the data for EDA with Pandas Profiling
def get_data(
    csv_path: Path
    ) -> Tuple[pd.DataFrame]:

    data = pd.read_csv(csv_path)

    return data
# Forest Cover Type Prediction - Capstone Rroject RS School ML Intro cource

###### tags: `Python, Poetry`

> This package uses Forest Type Prediction dataset ➜ [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data)  :evergreen_tree: 
> 
> Make sure your machine runs Python 3.9 and [Poetry](https://python-poetry.org/) :computer: 

## :memo: How to use this package?

### Step 1: Clone this repository to your machine

> You can check here how to clone repositories from GitHub ➜ [GitHub Docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) 

:rocket: 

### Step 2: Download and save locally the dataset

Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data), save csv locally (default path is data/train.csv in repository's root).

==ML_final_project contains two folders: "forest" and "screenshots". The "forest folder" contains package and is used as root.==

### Step 3: Run Python 3.9 and Poetry

Make sure your machine runs Python 3.9 and [Poetry](https://python-poetry.org/)

> You can check here how to switch between Python versions ➜ [pyenv](https://realpython.com/intro-to-pyenv/#installing-pyenv) 

:rocket: 

### Step 4: Install the project dependencies

Run this and following commands in a terminal, from the root of a cloned repository (forest):

- Install poetry package without development dependencies:
```python=1
poetry install --no-dev
```
- Run train:
```python=2
poetry run train -d <path to csv with data> -s <path to save trained model>
```
- Configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```python=3
poetry run train --help
```
- You can also use poetry scripts to execute train (default path) or get EDA (Pandas Profiling Report):
```python=4
poetry run profile_report
poetry run train
```
- Track your experiments results in MLflow UI:
```python=6
poetry run mlflow ui
```

![](https://i.imgur.com/MuyCueq.png)


> The same information about experiments is stored locally in mlruns folder:
> 
> ![](https://i.imgur.com/kensYPR.png)


### Step 5: Install development dependencies (Optional)

You need to install Development dependencies to run tests, format code and pass typeckeching before committing to GitHub.

- Install all requirements (including dev requirements) to poetry environment:
```python=7
poetry install
```

Now you can use additional developer tools.

- Run tests
```python=8
poetry run pytest
```
- Run tests
```python=9
poetry 
```
- Run tests
```python=10
poetry 
```

---


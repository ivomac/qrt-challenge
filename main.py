#!/usr/bin/env python
"""Stock Return Prediction by QRT.
This script is based on the provided benchmark template.
It trains a RandomForestClassifier on the training data with altered features.
The current parameters achieved 52.06% accuracy in validation and 51.75% in test.
(username:ivomac)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Global variables

## Flag to train the model and generate a submission
TRAINING = True

## Flag to print info and save it to a file
LOGGING = True

DATA_PATH = Path("./data")
OUTPUT_PATH = Path("./out")
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)


## Parameters

### smaller datatype for hopefully faster training
DATATYPE = np.float32

### Data cleaning: "zero", "drop" or "mean"
### "zero" replaces NaNs with 0
### "drop" removes rows with NaNs
### "mean" replaces NaNs with the mean of the column
NANS = "mean"

### Time aggregate quantities
### For each time-series feature, a transformation is applied
### and the result added to the dataframe
TIME_TRANSFORM = {
    "RET": [
        "std",
        "min",
    ],
    "VOLUME": [
        "std",
        "min",
    ],
}

### Group features
### Format: (target, shifts, stats, group)
### target: the feature to aggregate
### shifts: the time shift of the target: example, RET_{shift}
### stats: the aggregation functions to apply on each group
### group: the columns to group by
### For example, the first line will add
### the std of RET_1 for each DATE
GROUP_FEATURES = [
    ("RET", [1], ["std"], ["DATE"]),
    ("VOLUME", [1, 2], ["mean", "min"], ["SECTOR", "DATE"]),
    ("RET", [1], ["mean", "max", "min"], ["SECTOR", "DATE"]),
    ("RET", [1], ["max", "min"], ["SUB_INDUSTRY", "DATE"]),
]

### Feature selection
### Number of returns and volumes to keep
RETURNS_TO_KEEP = 4
VOLUMES_TO_KEEP = 1

### Training parameters
N_SPLITS = 4
RF_PARAMS = {
    "class_weight": "balanced",
    "n_estimators": 100,
    "max_depth": 2**2,
    "min_samples_split": 2**3,
    "min_samples_leaf": 2**2,
    "random_state": 0,
    "n_jobs": -1,
}


def main():
    # Load and clean data
    df, dates = load_data()

    # Feature Engineering
    ## Keep only some returns and volumes
    ## Add time-based and group-based features
    features = [f"RET_{i+1}" for i in range(RETURNS_TO_KEEP)]
    features += [f"VOLUME_{i+1}" for i in range(VOLUMES_TO_KEEP)]
    features += add_time_features(df)
    features += add_group_features(df)

    ## Sanity check: assert no NaN values
    sanity_check(df)

    if TRAINING:
        log_parameters()

        models, scores = train_model(df, dates, features)

        mean = log_accuracy(scores)
        log_importance(models, features)

        time = pd.Timestamp.now().strftime("%m.%d-%H:%M")
        out_name = f"acc_{mean:.2f}_{time}"

        user = input("Generate predictions on test? [y/N]")
        if user.lower() == "y":
            generate_submission(df, features, out_name)

        save_log(out_name)
    else:
        log_correlations(df, features)
    return


def load_data():
    """Load the data from the csv files and deal with NANs."""

    x_train = pd.read_csv(DATA_PATH / "x_train.csv", index_col="ID", dtype=DATATYPE)
    y_train = pd.read_csv(DATA_PATH / "y_train.csv", index_col="ID", dtype=DATATYPE)

    df = {}
    df["train"] = pd.concat([x_train, y_train], axis=1)

    df["test"] = pd.read_csv(DATA_PATH / "x_test.csv", index_col="ID", dtype=DATATYPE)

    dates = {}
    dates["train"] = df["train"]["DATE"]
    dates["test"] = df["test"]["DATE"]

    for data in df.values():
        if NANS == "zero":
            data.fillna(0, inplace=True)
        elif NANS == "drop":
            data.dropna(inplace=True)
        elif NANS == "mean":
            data.fillna(data.mean(), inplace=True)

    return df, dates


def add_time_features(df):
    """Add time-based features to the dataframes.
    For example, the mean of the 20 values of RETURNS/VOLUMES.
    """
    new_features = set()
    for feature, transforms in TIME_TRANSFORM.items():
        for transform in transforms:
            name = f"{feature}_{transform}"
            new_features.add(name)
            for data in df.values():
                base = data[[f"{feature}_{i+1}" for i in range(20)]]
                data[name] = base.agg(transform, axis=1)
    return list(new_features)


def add_group_features(df):
    """Add group-based features to the dataframes.
    For example, the mean of RET_1 on each DAY and SECTOR.
    """
    new_features = set()
    for target, shifts, stats, group in GROUP_FEATURES:
        tmp_name = "_".join(group)
        for shift in shifts:
            for stat in stats:
                feat = f"{target}_{shift}"
                name = f"{feat}_{tmp_name}_{stat}"
                new_features.add(name)
                for data in df.values():
                    data[name] = data.groupby(group)[feat].transform(stat)
    return list(new_features)


def sanity_check(df):
    """Check for NaN values in the dataframes.
    Also copy the dataframe to avoid fragmentation/performance problems.
    """
    for k, data in df.items():
        for col in data.columns:
            count = data[col].isna().sum()
            assert count == 0, f"{col} has {count} NaN values"
        df[k] = data.copy()


def train_model(df, dates, features):
    """Train a RandomForestClassifier on the training data."""

    scores = []
    models = []

    # split the data into train and validation sets
    splits = KFold(n_splits=N_SPLITS, random_state=0, shuffle=True).split(dates["train"].unique())

    for split in splits:
        ldf = {}
        for ids, name in zip(split, ["train", "validation"]):
            data = df["train"].loc[dates["train"].isin(ids)]
            ldf[name] = (data[features], data["RET"])

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(*ldf["train"])
        models.append(model)

        y_pred = model.predict(ldf["validation"][0])

        score = accuracy_score(ldf["validation"][1], y_pred)
        scores.append(score)

    return models, scores


def generate_submission(df, features, out_name):
    """Generate a submission file with the predictions on the test split."""

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(df["train"][features], df["train"]["RET"])
    y_pred = model.predict(df["test"][features])
    submission = pd.Series(y_pred)
    submission.index = df["test"].index
    submission.name = "RET"
    submission.to_csv(OUTPUT_PATH / f"{out_name}.csv", index=True, header=True)
    return


LOG = []


def log_parameters():
    """Log the parameters used for the training."""

    log("Parameters:")
    log(f"nans: {NANS}")
    log(f"time_transform: {TIME_TRANSFORM}")
    log(f"group_features: {GROUP_FEATURES}")
    log(f"returns_to_keep: {RETURNS_TO_KEEP}")
    log(f"volumes_to_keep: {VOLUMES_TO_KEEP}")
    log(f"n_splits: {N_SPLITS}")
    log(f"rf_params: {RF_PARAMS}\n")
    return


def log_accuracy(scores):
    """Log the accuracy of the model on the validation set."""

    for i, score in enumerate(scores):
        log(f"Fold {i+1} - Accuracy: {score*100:.2f}%")
    mean = np.mean(scores) * 100
    std = np.std(scores) * 100
    low, up = mean - std, mean + std
    log(f"Accuracy: {mean:.2f}% [{low:.2f} ; {up:.2f}] (+- {std:.2f})\n")
    return mean


def log_importance(models, features):
    """Log the feature importance of the models."""

    df_importance = pd.DataFrame([model.feature_importances_ for model in models], columns=features)
    importance = df_importance.mean()
    if isinstance(importance, pd.Series):
        importance = importance.sort_values(ascending=False)
    log("Feature importance:")
    log(str(importance) + "\n")
    return


def log_correlations(df, features):
    """Log the correlation between features and the target."""

    log("Correlation with returns:")
    corr = df["train"][features].corrwith(df["train"]["RET"]).sort_values(ascending=False, key=abs)
    for name, val in corr.items():
        if abs(val) < 0.004:
            log(f"{name}:\n{val:.2e}")

    log("\nCorrelation between features:")
    corr = df["train"][features].corr()

    corrs = []
    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1 :]:
            val = corr[col1][col2]
            corrs.append((col1, col2, val))
    corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    for col1, col2, val in corrs:
        if abs(val) > 0.4:
            log(f"{col1} - {col2}:\n{val:.2f}")
    return


def save_log(out_name):
    """Save the log to a file."""
    if LOGGING:
        with open(OUTPUT_PATH / f"{out_name}.txt", "w") as f:
            f.write("\n".join(LOG))
    return


def log(s):
    if LOGGING:
        LOG.append(s)
        print(s)


main()

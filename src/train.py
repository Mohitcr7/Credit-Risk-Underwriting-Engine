"""Train the probability-of-default model.

Evaluation choices, and why:
- Out-of-time split (train pre-2016, validate 2016+): credit models are
  deployed on future applicants, so random K-fold overstates performance
  when the borrower mix shifts over time.
- ROC-AUC for ranking power, PR-AUC because defaults are the minority class,
  Brier score + calibration because underwriting needs the *probability*
  to be right, not just the ordering — expected loss = PD x exposure.

Run:  python -m src.train
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src import config, features

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "metric": ["auc"],
    "verbosity": -1,
    "seed": 42,
}


def temporal_split(df: pd.DataFrame):
    split = pd.Timestamp(config.SPLIT_DATE)
    train = df[df["issue_d"] < split]
    valid = df[df["issue_d"] >= split]
    print(f"Train: {len(train):,} loans (< {config.SPLIT_DATE}), "
          f"default rate {train[config.TARGET].mean():.1%}")
    print(f"Valid: {len(valid):,} loans (>= {config.SPLIT_DATE}), "
          f"default rate {valid[config.TARGET].mean():.1%}")
    return train, valid


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    feat_cols = features.feature_columns()
    train, valid = temporal_split(df)

    dtrain = lgb.Dataset(train[feat_cols], label=train[config.TARGET])
    dvalid = lgb.Dataset(valid[feat_cols], label=valid[config.TARGET], reference=dtrain)

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    pd_valid = model.predict(valid[feat_cols], num_iteration=model.best_iteration)
    y = valid[config.TARGET].to_numpy()

    metrics = {
        "n_train": len(train),
        "n_valid": len(valid),
        "valid_default_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, pd_valid)),
        "pr_auc": float(average_precision_score(y, pd_valid)),
        "brier": float(brier_score_loss(y, pd_valid)),
        "best_iteration": model.best_iteration,
    }

    # Calibration table: predicted PD vs realized default rate by decile.
    deciles = pd.qcut(pd_valid, 10, labels=False, duplicates="drop")
    calib = (
        pd.DataFrame({"pred": pd_valid, "actual": y, "decile": deciles})
        .groupby("decile")
        .agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"), n=("actual", "size"))
    )
    print("\nCalibration by predicted-PD decile:")
    print(calib.to_string(float_format=lambda x: f"{x:.4f}"))

    config.MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(str(config.MODEL_DIR / "pd_model.txt"))
    (config.MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.MODEL_DIR / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    calib.to_csv(config.MODEL_DIR / "calibration.csv")

    print("\nValidation metrics (out-of-time, 2016+ vintages):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSaved model + metrics to {config.MODEL_DIR}/")


if __name__ == "__main__":
    main()

"""Recalibrate raw model scores with isotonic regression.

The training data (pre-2016) has a lower default rate than later vintages,
so raw PDs systematically underestimate risk (see models/calibration.csv).
Production fix: periodically refit a monotonic mapping raw-score -> PD on
the most recent *resolved* vintage. Here:

  fit  on 2016 loans (out-of-time w.r.t. training)
  eval on 2017+ loans (out-of-time w.r.t. both training AND calibration)

Isotonic regression is monotonic, so ranking metrics (AUC) are unchanged;
only probability quality (Brier, calibration error) improves.

Run:  python -m src.calibrate
"""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src import config, features

CALIB_START = "2016-01-01"
CALIB_END = "2017-01-01"


def decile_table(pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    deciles = pd.qcut(pred, 10, labels=False, duplicates="drop")
    return (
        pd.DataFrame({"pred": pred, "actual": y, "decile": deciles})
        .groupby("decile")
        .agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
    )


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    model = lgb.Booster(model_file=str(config.MODEL_DIR / "pd_model.txt"))
    feat_cols = features.feature_columns()

    calib = df[(df["issue_d"] >= CALIB_START) & (df["issue_d"] < CALIB_END)]
    evald = df[df["issue_d"] >= CALIB_END]
    print(f"Calibration set (2016): {len(calib):,} loans, "
          f"default rate {calib[config.TARGET].mean():.1%}")
    print(f"Evaluation set (2017+): {len(evald):,} loans, "
          f"default rate {evald[config.TARGET].mean():.1%}")

    raw_calib = model.predict(calib[feat_cols])
    raw_eval = model.predict(evald[feat_cols])
    y_eval = evald[config.TARGET].to_numpy()

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_calib, calib[config.TARGET])
    cal_eval = iso.predict(raw_eval)

    print(f"\nBrier on 2017+:  raw {brier_score_loss(y_eval, raw_eval):.4f}"
          f"  ->  calibrated {brier_score_loss(y_eval, cal_eval):.4f}")

    print("\nCalibration by decile on 2017+ (raw vs calibrated):")
    t = decile_table(raw_eval, y_eval).join(
        decile_table(cal_eval, y_eval), lsuffix="_raw", rsuffix="_cal"
    )
    print(t.to_string(float_format=lambda x: f"{x:.4f}"))

    joblib.dump(iso, config.MODEL_DIR / "calibrator.pkl")
    print(f"\nSaved {config.MODEL_DIR / 'calibrator.pkl'}")
    print("Re-run `python -m src.business` so the policy threshold is chosen "
          "on calibrated PDs (business.py applies the calibrator when present).")


if __name__ == "__main__":
    main()

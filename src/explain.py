"""SHAP explainability: global feature importance + per-loan reason codes.

Regulation (ECOA in the US, similar rules elsewhere) requires lenders to
give applicants specific reasons for adverse decisions. `reason_codes()`
produces exactly that from SHAP values, and is reused by the scoring API.

Run:  python -m src.explain   (writes global summary plot to reports/)
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src import config, features


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(config.MODEL_DIR / "pd_model.txt"))


def reason_codes(model: lgb.Booster, X: pd.DataFrame, top_n: int = 4) -> list[list[dict]]:
    """Top-N SHAP drivers per row, formatted for humans and for the agent.

    Returns, per row, a list of {feature, value, shap, direction} dicts
    sorted by |shap| descending. direction is 'increases risk' / 'decreases risk'.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # older shap returns [neg, pos]
        shap_values = shap_values[1]

    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(shap_values[i]))[:top_n]
        row = []
        for j in order:
            val = X.iloc[i, j]
            row.append({
                "feature": X.columns[j],
                "value": None if pd.isna(val) else (str(val) if not np.isreal(val) else float(val)),
                "shap": float(shap_values[i, j]),
                "direction": "increases risk" if shap_values[i, j] > 0 else "decreases risk",
            })
        out.append(row)
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(config.PROCESSED_PARQUET)
    valid = df[df["issue_d"] >= pd.Timestamp(config.SPLIT_DATE)]
    sample = valid.sample(20_000, random_state=42)
    feat_cols = features.feature_columns()
    model = load_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[feat_cols])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    config.REPORTS_DIR.mkdir(exist_ok=True)
    shap.summary_plot(shap_values, sample[feat_cols], show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_summary.png", dpi=150)
    print(f"Wrote {config.REPORTS_DIR / 'shap_summary.png'}")

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_cols)
    mean_abs.sort_values(ascending=False).to_csv(
        config.REPORTS_DIR / "global_importance.csv", header=["mean_abs_shap"]
    )
    print(f"Wrote {config.REPORTS_DIR / 'global_importance.csv'}")
    print("\nTop 10 global drivers:")
    print(mean_abs.sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()

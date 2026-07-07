"""Turn model probabilities into an approval policy.

A PD model alone doesn't make decisions. This module sweeps approval
thresholds and computes expected portfolio profit on the validation set:

  profit(loan) = interest earned            if repaid
  loss(loan)   = -LGD x loan amount         if defaulted

LGD (loss given default) defaults to 0.65, i.e. lenders typically recover
~35% of a charged-off unsecured loan. The optimal threshold maximizes total
profit — which is NOT the threshold that maximizes F1 or accuracy. That gap
is the point of this analysis.

Run:  python -m src.business
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, features

LGD = 0.65


def expected_profit(df: pd.DataFrame, approve_mask: np.ndarray) -> float:
    """Realized profit of approving exactly the masked loans."""
    approved = df[approve_mask]
    # Interest actually earned over the term for repaid loans (simplified:
    # installment * n_months - principal), full LGD loss for defaults.
    n_months = approved["term"].astype(str).astype(int)
    interest = approved["installment"] * n_months - approved["loan_amnt"]
    profit = np.where(
        approved[config.TARGET] == 1,
        -LGD * approved["loan_amnt"],
        interest,
    )
    return float(profit.sum())


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    valid = df[df["issue_d"] >= pd.Timestamp(config.SPLIT_DATE)].copy()

    model = lgb.Booster(model_file=str(config.MODEL_DIR / "pd_model.txt"))
    feat_cols = features.feature_columns()
    valid["pd"] = model.predict(valid[feat_cols])

    rows = []
    for threshold in np.arange(0.05, 0.61, 0.025):
        mask = (valid["pd"] < threshold).to_numpy()
        rows.append({
            "threshold": round(float(threshold), 3),
            "approval_rate": float(mask.mean()),
            "default_rate_approved": float(valid.loc[mask, config.TARGET].mean()),
            "portfolio_profit_musd": expected_profit(valid, mask) / 1e6,
        })

    table = pd.DataFrame(rows)
    best = table.loc[table["portfolio_profit_musd"].idxmax()]

    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nProfit-optimal threshold: approve if PD < {best['threshold']}")
    print(f"  approval rate {best['approval_rate']:.1%}, "
          f"default rate among approved {best['default_rate_approved']:.1%}, "
          f"portfolio profit ${best['portfolio_profit_musd']:.0f}M")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(config.REPORTS_DIR / "threshold_analysis.csv", index=False)
    (config.MODEL_DIR / "policy.json").write_text(json.dumps({
        "approve_below_pd": float(best["threshold"]),
        "lgd_assumption": LGD,
        "expected_approval_rate": float(best["approval_rate"]),
    }, indent=2))
    print(f"\nSaved policy to {config.MODEL_DIR}/policy.json")


if __name__ == "__main__":
    main()

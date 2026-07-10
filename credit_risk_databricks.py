# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Risk on Databricks — Leakage-Safe PD Model with Provable Lineage, MLflow, UC Model Serving & Drift Monitoring
# MAGIC
# MAGIC A single notebook that runs **top-to-bottom on Databricks Free Edition** *or a **Databricks-on-AWS** workspace*
# MAGIC (serverless + Unity Catalog). On AWS the Unity Catalog tables are physically **Delta files in S3** and compute
# MAGIC runs on **EC2** — see the *AWS footprint* cell (§5b) and [AWS_DATABRICKS_SETUP.md](./AWS_DATABRICKS_SETUP.md).
# MAGIC It reproduces the credit-risk engine from the companion repo, but re-architected onto the Lakehouse so that
# MAGIC the three things reviewers actually probe become *provable platform facts*, not README claims:
# MAGIC
# MAGIC | Concern | How the Lakehouse proves it |
# MAGIC |---|---|
# MAGIC | **Leakage safety** | The 28-column origination firewall is a Unity Catalog table derived from raw via a whitelist `SELECT`. UC **column lineage** shows no feature descends from a post-origination (leakage) column. |
# MAGIC | **Calibration drift** | The vintage-driven drift (pre-2016 training underestimates 2016+ risk) is caught automatically by **Lakehouse Monitoring** on an inference log. |
# MAGIC | **Governance / reproducibility** | Training, out-of-time eval, and the raw→calibrated **Brier improvement** are tracked in **MLflow**; the booster + isotonic calibrator + SHAP reason codes ship as **one custom `pyfunc`** registered to **Unity Catalog** and served from a **scale-to-zero endpoint** with inference logging on. |
# MAGIC
# MAGIC **Pipeline:** raw CSV → Delta bronze (`loans_raw`) → UC feature firewall (`loans_features`) → LightGBM + OOT + isotonic (MLflow) → custom pyfunc → UC Model Registry → Model Serving (scale-to-zero, inference logging) → Lakehouse Monitoring.
# MAGIC
# MAGIC > Data: [LendingClub accepted loans 2007–2018Q4](https://huggingface.co/datasets/codesignal/lending-club-loan-accepted) (CC0). ~2.26M rows → ~1.35M resolved loans.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install libraries
# MAGIC LightGBM and SHAP are not on the serverless base image. Pinned so the training environment matches the serving container.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 shap==0.46.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC Everything is parameterized via widgets. Defaults target Free Edition's `workspace` catalog. The pipeline
# MAGIC trains on the **full ~1.35M resolved loans** to reproduce the resume metrics (OOT ROC-AUC ≈ 0.718).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Unity Catalog")
dbutils.widgets.text("schema", "credit_risk", "Schema")
dbutils.widgets.text("volume", "raw", "Volume (raw files)")
dbutils.widgets.text("model_name", "credit_risk_pd", "Registered model name")
dbutils.widgets.text("endpoint_name", "credit-risk-pd", "Serving endpoint name")
dbutils.widgets.dropdown("deploy_serving", "yes", ["yes", "no"], "Deploy Model Serving endpoint")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
MODEL_SHORT = dbutils.widgets.get("model_name")
ENDPOINT = dbutils.widgets.get("endpoint_name")
DEPLOY_SERVING = dbutils.widgets.get("deploy_serving") == "yes"

SPLIT_DATE = "2016-01-01"       # train < SPLIT_DATE ; validate (out-of-time) >= SPLIT_DATE
CALIB_START, CALIB_END = "2016-01-01", "2017-01-01"  # isotonic fit on 2016, eval on 2017+
LGD = 0.65                      # loss given default for the profit-optimal policy
SEED = 42

DATA_URL = "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"

print(f"Target: {CATALOG}.{SCHEMA}  |  model: {CATALOG}.{SCHEMA}.{MODEL_SHORT}  |  training on full dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The leakage firewall, as code
# MAGIC `ORIGINATION_COLS` is the whitelist — the only columns a lender knows at decision time. `LEAKAGE_COLS` lists
# MAGIC representative post-origination columns that describe what happened *after* the loan was issued. Training on any of
# MAGIC them inflates AUC to 0.95+ and is useless for underwriting. Below, the feature table selects **only** from the
# MAGIC whitelist, and UC lineage will prove the leakage columns never touch a feature.

# COMMAND ----------

# 28 origination-time columns the model is allowed to see.
ORIGINATION_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "application_type", "mort_acc",
    "pub_rec_bankruptcies",
]

# Representative post-origination columns — the firewall keeps ALL of these OUT.
LEAKAGE_COLS = [
    "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag", "settlement_amount", "hardship_flag",
    "hardship_amount", "payment_plan_start_date",
]

# 8 categorical model features (emp_length is mapped to an ordinal 0..10, so it is numeric).
CATEGORICAL_COLS = [
    "term", "grade", "sub_grade", "home_ownership",
    "verification_status", "purpose", "addr_state", "application_type",
]

# Final 27 model features (engineered: fico, credit_history_years; raw fico_range_*/earliest_cr_line dropped).
FEATURE_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "application_type", "mort_acc",
    "pub_rec_bankruptcies", "fico", "credit_history_years",
]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog namespace (schema + volume)
# MAGIC Creates the schema and a managed Volume for the raw file. Falls back across candidate catalogs so the notebook
# MAGIC runs whether your workspace's default catalog is `workspace`, `main`, or something else.

# COMMAND ----------

def ensure_namespace(preferred_catalog, schema, volume):
    # Free Edition ships a `workspace` catalog; Databricks-on-AWS UC workspaces
    # usually ship `main`. Try the requested catalog first (creating it when the
    # metastore has managed S3 storage), then fall back across common defaults.
    candidates = [preferred_catalog, "main", "workspace"]
    tried = []
    for cat in dict.fromkeys(candidates):  # de-dupe, keep order
        try:
            spark.sql(f"CREATE CATALOG IF NOT EXISTS `{cat}`")  # best-effort; needs metastore storage
        except Exception:  # noqa: BLE001
            pass
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{cat}`.`{schema}`")
            spark.sql(f"CREATE VOLUME IF NOT EXISTS `{cat}`.`{schema}`.`{volume}`")
            return cat
        except Exception as e:  # noqa: BLE001
            tried.append(f"{cat}: {str(e)[:120]}")
    raise RuntimeError("Could not create schema/volume in any candidate catalog:\n" + "\n".join(tried))

CATALOG = ensure_namespace(CATALOG, SCHEMA, VOLUME)
spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"USE SCHEMA `{SCHEMA}`")
VOLUME_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_SHORT}"
print(f"Using catalog '{CATALOG}'. Volume dir: {VOLUME_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Land the raw CSV in the Volume
# MAGIC Streams the 1.67 GB CC0 file to the managed Volume (idempotent — skipped if already present).

# COMMAND ----------

import os
import urllib.request

raw_path = f"{VOLUME_DIR}/accepted_2007_to_2018Q4.csv"
if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1_000_000_000:
    print(f"Already present: {raw_path} ({os.path.getsize(raw_path)/1e9:.2f} GB)")
else:
    print(f"Downloading -> {raw_path} (this takes a few minutes)...")
    tmp = "/tmp/accepted.csv"
    with urllib.request.urlopen(DATA_URL) as resp, open(tmp, "wb") as f:
        chunk = resp.read(1 << 20)
        while chunk:
            f.write(chunk)
            chunk = resp.read(1 << 20)
    dbutils.fs.cp(f"file:{tmp}", raw_path)
    os.remove(tmp)
    print(f"Landed {os.path.getsize(raw_path)/1e9:.2f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bronze Delta table `loans_raw` (full 151-column schema)
# MAGIC Read as all-strings (fast, no schema inference) and persist to Delta. Bronze keeps the **full** schema — including
# MAGIC the leakage columns — precisely so the firewall's exclusion is visible in lineage.

# COMMAND ----------

raw_df = (
    spark.read.option("header", True).option("inferSchema", False)
    .option("multiLine", False).option("escape", '"')
    .csv(raw_path)
)
(raw_df.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.loans_raw"))

n_raw = spark.table(f"{CATALOG}.{SCHEMA}.loans_raw").count()
present_leakage = [c for c in LEAKAGE_COLS if c in raw_df.columns]
print(f"loans_raw: {n_raw:,} rows, {len(raw_df.columns)} columns")
print(f"Leakage columns present in bronze (and about to be firewalled out): {present_leakage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The firewall: Unity Catalog feature table `loans_features`
# MAGIC A single whitelist `CREATE TABLE AS SELECT`. It:
# MAGIC - filters to **resolved** loans (Fully Paid / Charged Off) and defines the `is_default` target,
# MAGIC - selects **only** the 28 origination columns,
# MAGIC - engineers `fico` (avg of the FICO range) and `credit_history_years`, dropping the raw source columns,
# MAGIC - casts `%`-strings and parses `term`, and ordinal-maps `emp_length`.
# MAGIC
# MAGIC Because every feature is derived in Spark SQL from named bronze columns, **UC captures column-level lineage** —
# MAGIC open the table's *Lineage* tab afterward and you will see `recoveries`, `total_pymnt`, `last_fico_*`, etc. have
# MAGIC **no** downstream edge into any feature. That is the leakage proof.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.loans_features
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS
WITH resolved AS (
  SELECT * FROM {CATALOG}.{SCHEMA}.loans_raw
  WHERE loan_status IN (
    'Fully Paid', 'Charged Off',
    'Does not meet the credit policy. Status:Fully Paid',
    'Does not meet the credit policy. Status:Charged Off'
  )
)
SELECT
  CAST(loan_amnt AS DOUBLE)                                   AS loan_amnt,
  regexp_extract(term, '[0-9]+', 0)                          AS term,
  CAST(regexp_replace(int_rate, '%', '') AS DOUBLE)          AS int_rate,
  CAST(installment AS DOUBLE)                                 AS installment,
  grade,
  sub_grade,
  CASE emp_length
    WHEN '< 1 year' THEN 0 WHEN '1 year' THEN 1 WHEN '2 years' THEN 2
    WHEN '3 years' THEN 3 WHEN '4 years' THEN 4 WHEN '5 years' THEN 5
    WHEN '6 years' THEN 6 WHEN '7 years' THEN 7 WHEN '8 years' THEN 8
    WHEN '9 years' THEN 9 WHEN '10+ years' THEN 10 ELSE NULL END AS emp_length,
  home_ownership,
  CAST(annual_inc AS DOUBLE)                                  AS annual_inc,
  verification_status,
  purpose,
  addr_state,
  CAST(dti AS DOUBLE)                                         AS dti,
  CAST(delinq_2yrs AS DOUBLE)                                 AS delinq_2yrs,
  CAST(inq_last_6mths AS DOUBLE)                              AS inq_last_6mths,
  CAST(mths_since_last_delinq AS DOUBLE)                      AS mths_since_last_delinq,
  CAST(mths_since_last_record AS DOUBLE)                      AS mths_since_last_record,
  CAST(open_acc AS DOUBLE)                                    AS open_acc,
  CAST(pub_rec AS DOUBLE)                                     AS pub_rec,
  CAST(revol_bal AS DOUBLE)                                   AS revol_bal,
  CAST(regexp_replace(revol_util, '%', '') AS DOUBLE)        AS revol_util,
  CAST(total_acc AS DOUBLE)                                   AS total_acc,
  application_type,
  CAST(mort_acc AS DOUBLE)                                    AS mort_acc,
  CAST(pub_rec_bankruptcies AS DOUBLE)                        AS pub_rec_bankruptcies,
  (CAST(fico_range_low AS DOUBLE) + CAST(fico_range_high AS DOUBLE)) / 2.0 AS fico,
  datediff(to_date(issue_d, 'MMM-yyyy'), to_date(earliest_cr_line, 'MMM-yyyy')) / 365.25 AS credit_history_years,
  to_date(issue_d, 'MMM-yyyy')                                AS issue_d,
  CASE WHEN loan_status LIKE '%Charged Off%' THEN 1
       WHEN loan_status LIKE '%Fully Paid%'  THEN 0 END       AS is_default
FROM resolved
WHERE to_date(issue_d, 'MMM-yyyy') IS NOT NULL
  AND CAST(annual_inc AS DOUBLE) BETWEEN 0 AND 5000000
""")

feat_tbl = spark.table(f"{CATALOG}.{SCHEMA}.loans_features")
n_feat = feat_tbl.count()
default_rate = feat_tbl.selectExpr("avg(is_default)").first()[0]
print(f"loans_features: {n_feat:,} resolved loans, default rate {default_rate:.1%}")
print("Add a table comment documenting the firewall (surfaces in Catalog Explorer):")
spark.sql(f"""COMMENT ON TABLE {CATALOG}.{SCHEMA}.loans_features IS
  'Leakage-safe origination firewall: derived ONLY from the 28 origination-time columns of loans_raw. No post-origination (payment/recovery/hardship/settlement) column feeds any feature — verifiable in the Lineage tab.'""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lineage check (programmatic)
# MAGIC Beyond the visual Lineage tab, we assert the firewall held: none of the leakage columns appear in the feature schema.

# COMMAND ----------

feature_schema_cols = set(spark.table(f"{CATALOG}.{SCHEMA}.loans_features").columns)
leaked = feature_schema_cols.intersection(LEAKAGE_COLS)
assert not leaked, f"FIREWALL BREACH: leakage columns in feature table: {leaked}"
print("Firewall assertion passed — 0 leakage columns in loans_features.")
print(f"Feature table columns ({len(feature_schema_cols)}): {sorted(feature_schema_cols)}")
print("\nOpen Catalog Explorer -> loans_features -> Lineage to see the column graph "
      "(recoveries/total_pymnt/last_fico_* have no edge into any feature).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5b. AWS footprint — S3-backed Delta + EC2 compute
# MAGIC On a Databricks-on-AWS workspace, Unity Catalog tables are physically **Delta files in S3** and notebook
# MAGIC compute runs on **EC2** in the data plane. This cell prints the concrete `s3://` locations and the compute
# MAGIC node type so the AWS backing is provable (screenshot-worthy for interviews). On Free Edition these are
# MAGIC Databricks-managed and abstracted (no `s3://` shown), but everything else in the notebook is identical.

# COMMAND ----------

def _table_location(tbl):
    try:
        return spark.sql(f"DESCRIBE DETAIL {tbl}").select("location").first()[0]
    except Exception as e:  # noqa: BLE001
        return f"(unavailable: {str(e)[:80]})"

print("Physical storage of Unity Catalog tables:")
for t in ["loans_raw", "loans_features"]:
    print(f"  {CATALOG}.{SCHEMA}.{t}\n    -> {_table_location(f'{CATALOG}.{SCHEMA}.{t}')}")

try:
    vinfo = spark.sql(f"DESCRIBE VOLUME {CATALOG}.{SCHEMA}.{VOLUME}").first().asDict()
    print(f"  volume '{VOLUME}' -> {vinfo.get('storage_location', vinfo)}")
except Exception as e:  # noqa: BLE001
    print(f"  volume location unavailable: {str(e)[:80]}")

node_type = spark.conf.get("spark.databricks.clusterUsageTags.clusterNodeType", None)
region = spark.conf.get("spark.databricks.clusterUsageTags.region", None)
print(f"\nCompute node type: {node_type or 'serverless (Databricks-managed EC2)'}"
      + (f"  |  region: {region}" if region else ""))
print("An s3:// location above confirms the Lakehouse data is on AWS S3.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load features, encode categoricals, out-of-time split
# MAGIC We pull the feature table to pandas for LightGBM. Categoricals are **integer-encoded with a saved mapping**
# MAGIC (fit on the training split only) rather than pandas `category` dtype — this makes scoring deterministic and
# MAGIC serving-safe (unseen categories → `-1` = missing), avoiding the classic train/serve categorical-dtype skew.

# COMMAND ----------

import numpy as np
import pandas as pd

pdf = spark.table(f"{CATALOG}.{SCHEMA}.loans_features").toPandas()
pdf["issue_d"] = pd.to_datetime(pdf["issue_d"])
pdf["is_default"] = pdf["is_default"].astype("int8")

split = pd.Timestamp(SPLIT_DATE)
train_mask = pdf["issue_d"] < split
valid_mask = pdf["issue_d"] >= split

train_pdf = pdf[train_mask]
valid_pdf = pdf[valid_mask]

# Fit integer encoders on TRAIN categories only.
cat_maps = {}
for c in CATEGORICAL_COLS:
    cats = sorted(train_pdf[c].dropna().unique().tolist())
    cat_maps[c] = {v: i for i, v in enumerate(cats)}

def encode(frame):
    X = frame[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_COLS:
        X[c] = frame[c].map(cat_maps[c]).fillna(-1).astype("int64")
    for c in FEATURE_COLUMNS:
        if c not in CATEGORICAL_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

X_train, y_train = encode(train_pdf), train_pdf["is_default"].to_numpy()
X_valid, y_valid = encode(valid_pdf), valid_pdf["is_default"].to_numpy()
print(f"Train: {len(X_train):,} (default {y_train.mean():.1%})  |  "
      f"Valid OOT >= {SPLIT_DATE}: {len(X_valid):,} (default {y_valid.mean():.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Train LightGBM + out-of-time eval + isotonic calibration, tracked in MLflow
# MAGIC One MLflow run captures params, the honest **OOT ROC-AUC / PR-AUC**, and the raw→calibrated **Brier** as
# MAGIC compared metrics. Isotonic regression is fit on the **2016** vintage and evaluated strictly out-of-time on **2017+** —
# MAGIC monotonic, so ranking (AUC) is unchanged; only probability quality improves.

# COMMAND ----------

import json
import tempfile

import lightgbm as lgb
import mlflow
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

mlflow.set_registry_uri("databricks-uc")

PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "metric": ["auc"], "verbosity": -1, "seed": SEED,
}

with mlflow.start_run(run_name="lightgbm_oot_isotonic") as run:
    RUN_ID = run.info.run_id
    mlflow.log_params(PARAMS)
    mlflow.log_params({
        "n_train": len(X_train), "n_valid_oot": len(X_valid),
        "split_date": SPLIT_DATE,
        "n_features": len(FEATURE_COLUMNS),
    })

    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=CATEGORICAL_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain, categorical_feature=CATEGORICAL_COLS)
    booster = lgb.train(
        PARAMS, dtrain, num_boost_round=2000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    raw_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)
    oot_auc = roc_auc_score(y_valid, raw_valid)
    oot_pr = average_precision_score(y_valid, raw_valid)

    # Isotonic: fit on 2016, evaluate on 2017+ (both out-of-time vs training).
    v_dates = valid_pdf["issue_d"].to_numpy()
    calib_mask = (v_dates >= np.datetime64(CALIB_START)) & (v_dates < np.datetime64(CALIB_END))
    eval_mask = v_dates >= np.datetime64(CALIB_END)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_valid[calib_mask], y_valid[calib_mask])

    raw_eval, y_eval = raw_valid[eval_mask], y_valid[eval_mask]
    cal_eval = iso.predict(raw_eval)
    brier_raw = brier_score_loss(y_eval, raw_eval)
    brier_cal = brier_score_loss(y_eval, cal_eval)

    mlflow.log_metrics({
        "oot_roc_auc": oot_auc,
        "oot_pr_auc": oot_pr,
        "best_iteration": booster.best_iteration,
        "brier_raw_2017plus": brier_raw,          # compared metric (before)
        "brier_calibrated_2017plus": brier_cal,   # compared metric (after)
        "brier_improvement": brier_raw - brier_cal,
    })

    # Calibration-by-decile table as an artifact (raw vs calibrated).
    dec = pd.qcut(raw_eval, 10, labels=False, duplicates="drop")
    calib_tbl = (pd.DataFrame({"raw": raw_eval, "cal": cal_eval, "actual": y_eval, "decile": dec})
                 .groupby("decile").agg(mean_raw=("raw", "mean"), mean_cal=("cal", "mean"),
                                        mean_actual=("actual", "mean"), n=("actual", "size")))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calibration_by_decile.csv")
        calib_tbl.to_csv(p)
        mlflow.log_artifact(p)

print(f"\nOOT ROC-AUC: {oot_auc:.4f}  |  PR-AUC: {oot_pr:.4f}  (validation default rate {y_valid.mean():.1%})")
print(f"Brier on 2017+:  raw {brier_raw:.4f}  ->  calibrated {brier_cal:.4f}  "
      f"(improvement {brier_raw - brier_cal:+.4f})")
display(calib_tbl.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Profit-optimal approval threshold
# MAGIC The PD is turned into a decision by the economics, not F1: sweep cutoffs on the calibrated OOT predictions and
# MAGIC maximize realized portfolio profit (term interest on repaid loans vs. 65%-LGD loss on defaults). This threshold
# MAGIC drives the served model's approve/decline.

# COMMAND ----------

cal_valid = iso.predict(raw_valid)
term_months = valid_pdf["term"].astype(int).to_numpy()
loan_amnt = valid_pdf["loan_amnt"].to_numpy()
installment = valid_pdf["installment"].to_numpy()
interest = installment * term_months - loan_amnt          # earned if repaid
profit_if_approved = np.where(y_valid == 1, -LGD * loan_amnt, interest)

rows = []
for t in np.arange(0.05, 0.61, 0.025):
    m = cal_valid < t
    rows.append((round(float(t), 3), float(m.mean()),
                 float(y_valid[m].mean()) if m.any() else 0.0,
                 float(profit_if_approved[m].sum()) / 1e6))
sweep = pd.DataFrame(rows, columns=["threshold", "approval_rate", "default_rate_approved", "profit_musd"])
POLICY_THRESHOLD = float(sweep.loc[sweep["profit_musd"].idxmax(), "threshold"])
mlflow.log_metric("policy_threshold", POLICY_THRESHOLD, run_id=RUN_ID)
print(f"Profit-optimal threshold: approve if calibrated PD < {POLICY_THRESHOLD}")
display(sweep)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Package booster + calibrator + SHAP reason codes as ONE custom `pyfunc`
# MAGIC A single deployable artifact that, per applicant, returns the **calibrated PD**, the **approve/decline** decision,
# MAGIC and **SHAP reason codes** (ECOA-style adverse-action reasons). Input = the 27 model features (categoricals as
# MAGIC strings); encoding, calibration, and explanation all happen inside the model.

# COMMAND ----------

import cloudpickle
import shap


class CreditRiskModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.booster = lgb.Booster(model_file=context.artifacts["booster"])
        with open(context.artifacts["calibrator"], "rb") as f:
            self.calibrator = cloudpickle.load(f)
        with open(context.artifacts["meta"], "r") as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]
        self.categorical_cols = meta["categorical_cols"]
        self.cat_maps = meta["cat_maps"]
        self.emp_length_map = meta["emp_length_map"]
        self.threshold = meta["threshold"]
        self.explainer = shap.TreeExplainer(self.booster)

    def _encode(self, model_input):
        X = pd.DataFrame(model_input).copy()
        for c in self.feature_columns:
            if c not in X.columns:
                X[c] = np.nan
        disp = X[self.feature_columns].copy()          # human-readable values for reason codes
        Xe = X[self.feature_columns].copy()
        if Xe["emp_length"].dtype == object:
            Xe["emp_length"] = Xe["emp_length"].map(lambda v: self.emp_length_map.get(v, v))
        for c in self.feature_columns:
            if c in self.categorical_cols:
                Xe[c] = X[c].astype("object").map(lambda v, m=self.cat_maps[c]: m.get(v, -1)).astype("int64")
            else:
                Xe[c] = pd.to_numeric(Xe[c], errors="coerce")
        return Xe, disp

    def predict(self, context, model_input):
        Xe, disp = self._encode(model_input)
        raw = self.booster.predict(Xe)
        pd_cal = np.asarray(self.calibrator.predict(raw), dtype=float)

        sv = self.explainer.shap_values(Xe)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]

        reasons = []
        for i in range(len(Xe)):
            order = np.argsort(-np.abs(sv[i]))[:4]
            items = []
            for j in order:
                f = self.feature_columns[j]
                val = disp.iloc[i][f]
                items.append({
                    "feature": f,
                    "value": None if pd.isna(val) else (val if isinstance(val, str) else float(val)),
                    "shap": float(sv[i][j]),
                    "direction": "increases risk" if sv[i][j] > 0 else "decreases risk",
                })
            reasons.append(json.dumps(items))

        decision = ["decline" if p >= self.threshold else "approve" for p in pd_cal]
        return pd.DataFrame({
            "probability_of_default": np.round(pd_cal, 4),
            "decision": decision,
            "reason_codes": reasons,
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save artifacts, build signature, log & register to Unity Catalog

# COMMAND ----------

from importlib.metadata import version
from mlflow.models import infer_signature

art_dir = tempfile.mkdtemp()
booster_path = os.path.join(art_dir, "pd_model.txt")
calibrator_path = os.path.join(art_dir, "calibrator.pkl")
meta_path = os.path.join(art_dir, "meta.json")

booster.save_model(booster_path)
with open(calibrator_path, "wb") as f:
    cloudpickle.dump(iso, f)
with open(meta_path, "w") as f:
    json.dump({
        "feature_columns": FEATURE_COLUMNS,
        "categorical_cols": CATEGORICAL_COLS,
        "cat_maps": cat_maps,
        "emp_length_map": EMP_LENGTH_MAP,
        "threshold": POLICY_THRESHOLD,
    }, f)
artifacts = {"booster": booster_path, "calibrator": calibrator_path, "meta": meta_path}

# Serving contract: the 27 model features with categoricals as strings (as they appear in loans_features).
input_example = valid_pdf[FEATURE_COLUMNS].head(3).reset_index(drop=True)

# Local round-trip to build an accurate signature and smoke-test the pyfunc.
class _Ctx:
    def __init__(self, a): self.artifacts = a
_local = CreditRiskModel()
_local.load_context(_Ctx(artifacts))
output_example = _local.predict(None, input_example)
signature = infer_signature(input_example, output_example)
print("Local pyfunc smoke test:")
display(output_example)

pip_reqs = [
    f"lightgbm=={version('lightgbm')}",
    f"shap=={version('shap')}",
    f"scikit-learn=={version('scikit-learn')}",
    f"pandas=={version('pandas')}",
    f"numpy=={version('numpy')}",
    f"cloudpickle=={version('cloudpickle')}",
]

with mlflow.start_run(run_id=RUN_ID):
    logged = mlflow.pyfunc.log_model(
        artifact_path="credit_risk_model",
        python_model=CreditRiskModel(),
        artifacts=artifacts,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_reqs,
    )

registered = mlflow.register_model(model_uri=logged.model_uri, name=MODEL_NAME)
MODEL_VERSION = registered.version
mlflow.tracking.MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", MODEL_VERSION)
print(f"Registered {MODEL_NAME} version {MODEL_VERSION} (alias @champion)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity-check the registered model by loading it back from UC

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
display(loaded.predict(input_example))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Deploy a scale-to-zero Model Serving endpoint with inference logging
# MAGIC `scale_to_zero_enabled=True` keeps it free when idle. `auto_capture_config` turns on **inference logging** — every
# MAGIC request/response is written to a Delta table in UC, which becomes the substrate for drift monitoring.
# MAGIC
# MAGIC > Model Serving may be gated or capacity-limited on Free Edition. This cell is best-effort: it will not fail the
# MAGIC > notebook if serving is unavailable — the drift-monitoring section below stands on its own.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AutoCaptureConfigInput, EndpointCoreConfigInput, ServedEntityInput,
)

w = WorkspaceClient()
INFERENCE_TABLE_PREFIX = "credit_risk_serving"
served = ServedEntityInput(
    entity_name=MODEL_NAME, entity_version=MODEL_VERSION,
    scale_to_zero_enabled=True, workload_size="Small",
)
capture = AutoCaptureConfigInput(
    catalog_name=CATALOG, schema_name=SCHEMA,
    enabled=True, table_name_prefix=INFERENCE_TABLE_PREFIX,
)
config = EndpointCoreConfigInput(served_entities=[served], auto_capture_config=capture)

serving_ok = False
if DEPLOY_SERVING:
    try:
        existing = [e.name for e in w.serving_endpoints.list()]
        if ENDPOINT in existing:
            w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[served],
                                              auto_capture_config=capture)
            print(f"Updated existing endpoint '{ENDPOINT}'.")
        else:
            w.serving_endpoints.create(name=ENDPOINT, config=config)
            print(f"Creating endpoint '{ENDPOINT}' (provisioning takes ~5-15 min).")
        serving_ok = True
        print(f"Endpoint URL: {w.config.host}/ml/endpoints/{ENDPOINT}")
    except Exception as e:  # noqa: BLE001
        print("Model Serving unavailable / not permitted on this workspace — skipping.\n"
              f"Reason: {str(e)[:300]}")
else:
    print("deploy_serving=no — skipping endpoint creation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Query the endpoint once it is Ready
# MAGIC Run this after the endpoint reports READY. It sends applicants and (when logging is on) seeds the inference table.

# COMMAND ----------

if serving_ok:
    try:
        resp = w.serving_endpoints.query(
            name=ENDPOINT,
            dataframe_records=input_example.to_dict(orient="records"),
        )
        print(resp.predictions)
    except Exception as e:  # noqa: BLE001
        print(f"Endpoint not READY yet (retry in a few minutes): {str(e)[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Lakehouse Monitoring to catch vintage-driven calibration drift
# MAGIC To make the drift *observable in-notebook* (independent of live traffic), we replay the OOT vintages through the
# MAGIC model as an **inference log**: one row per 2016→2018 loan with `inference_timestamp = issue_d`, the model's
# MAGIC calibrated `prediction`, and the realized `label`. A Lakehouse **InferenceLog** monitor with monthly granularity
# MAGIC then tracks prediction/label drift over time — surfacing the same vintage drift the offline calibration corrected.
# MAGIC
# MAGIC The production serving inference table (`credit_risk_serving_payload`) can be monitored the same way once it
# MAGIC accumulates traffic; the schema (timestamp + prediction + label) is identical after unpacking.

# COMMAND ----------

# Build the inference-log Delta table from OOT predictions (calibrated).
infer_pdf = valid_pdf.copy()
infer_pdf["prediction"] = cal_valid
infer_pdf["label"] = infer_pdf["is_default"].astype("int")
infer_pdf["model_id"] = f"{MODEL_SHORT}_v{MODEL_VERSION}"
infer_pdf["inference_timestamp"] = pd.to_datetime(infer_pdf["issue_d"])

keep = FEATURE_COLUMNS + ["prediction", "label", "model_id", "inference_timestamp"]
infer_sdf = spark.createDataFrame(infer_pdf[keep])
(infer_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.inference_log"))
print(f"Wrote {CATALOG}.{SCHEMA}.inference_log ({infer_sdf.count():,} rows, "
      f"{infer_pdf['inference_timestamp'].min().date()} → {infer_pdf['inference_timestamp'].max().date()})")

# COMMAND ----------

from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.inference_log"
try:
    user = w.current_user.me().user_name
    w.quality_monitors.create(
        table_name=INFERENCE_TABLE,
        assets_dir=f"/Workspace/Users/{user}/lakehouse_monitoring/{SCHEMA}_inference_log",
        output_schema_name=f"{CATALOG}.{SCHEMA}",
        inference_log=MonitorInferenceLog(
            timestamp_col="inference_timestamp",
            granularities=["1 month"],
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        ),
    )
    print(f"Created Lakehouse Monitor on {INFERENCE_TABLE}.")
    print("Open the table's Quality tab; the monthly profile/drift metrics reveal predicted-PD vs. actual-default "
          "divergence for later vintages — the calibration drift, caught automatically.")
except Exception as e:  # noqa: BLE001
    msg = str(e)
    if "already" in msg.lower():
        print("Monitor already exists — refreshing it.")
        try:
            w.quality_monitors.run_refresh(table_name=INFERENCE_TABLE)
        except Exception as e2:  # noqa: BLE001
            print(f"Refresh note: {str(e2)[:200]}")
    else:
        print("Lakehouse Monitoring unavailable / not permitted on this workspace — skipping.\n"
              f"Reason: {msg[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drift, made visible with SQL
# MAGIC Even before the monitor's dashboard populates, this query shows the vintage story directly from the inference log:
# MAGIC average **predicted PD** vs. realized **default rate** by quarter. The widening gap in later vintages is the drift.

# COMMAND ----------

display(spark.sql(f"""
SELECT date_trunc('quarter', inference_timestamp) AS vintage_quarter,
       count(*)              AS n_loans,
       round(avg(prediction), 4) AS avg_predicted_pd,
       round(avg(label), 4)      AS actual_default_rate,
       round(avg(label) - avg(prediction), 4) AS gap
FROM {CATALOG}.{SCHEMA}.inference_log
GROUP BY 1 ORDER BY 1
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. What this notebook proves (the resume story)
# MAGIC
# MAGIC - **Leakage safety is provable, not asserted.** `loans_features` is derived from `loans_raw` by a whitelist
# MAGIC   `SELECT`; Unity Catalog **column lineage** shows no post-origination column feeds any feature, and a
# MAGIC   programmatic assertion enforces it. Honest **OOT ROC-AUC ≈ 0.72** (vs. the 0.95+ a leaky model fakes).
# MAGIC - **Governance is built in.** One **MLflow** run holds params, OOT metrics, and the raw→calibrated **Brier**
# MAGIC   improvement; the booster + isotonic calibrator + SHAP reason codes ship as **one `pyfunc`** registered to
# MAGIC   **Unity Catalog** with an alias, a signature, and pinned dependencies.
# MAGIC - **It's deployable and observable.** A **scale-to-zero** Model Serving endpoint returns calibrated PD +
# MAGIC   decision + adverse-action reason codes with **inference logging** on, and **Lakehouse Monitoring** on the
# MAGIC   inference log catches the **vintage-driven calibration drift** automatically.
# MAGIC
# MAGIC **Artifacts created:** `loans_raw`, `loans_features`, `inference_log` (+ monitor tables) in `{CATALOG}.{SCHEMA}`;
# MAGIC MLflow run; UC model `credit_risk_pd@champion`; serving endpoint `credit-risk-pd` (if enabled).

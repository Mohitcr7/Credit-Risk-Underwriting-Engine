# Running the notebook on Databricks-on-AWS

[credit_risk_databricks.py](credit_risk_databricks.py) runs unchanged on a Databricks **AWS** workspace. The payoff:
Unity Catalog tables become physical **Delta files in Amazon S3** and notebook compute runs on **EC2** in the data
plane — so the project legitimately demonstrates *both* Databricks and AWS. The `§5b AWS footprint` cell prints the
`s3://` locations and the EC2 node type as proof.

> This is the same notebook you'd run on Free Edition — nothing here fakes AWS. The difference is purely the workspace
> it executes in: on AWS, the Lakehouse storage/compute are your account's S3 + EC2.

## Prerequisites

- A Databricks **AWS** workspace on the **Premium** (or Enterprise) tier — required for Unity Catalog + serverless +
  Model Serving. The 14-day trial qualifies.
- You are an **account admin** (or have `CREATE CATALOG`/`CREATE SCHEMA` + serverless privileges).

## Step 1 — Confirm a Unity Catalog metastore is attached (this is the S3 + IAM part)

Unity Catalog stores managed data in an **S3 bucket** accessed via an **IAM role**. Newer AWS workspaces auto-provision
a regional metastore; if yours already has a `main` catalog, you're done — skip to Step 2.

If no metastore is attached (Catalog Explorer shows none):
1. **S3 bucket** — create one, e.g. `s3://<you>-uc-metastore-<region>/`.
2. **IAM role** — create a role Databricks can assume to access that bucket (trust the Databricks account + your
   external ID; grant `s3:GetObject/PutObject/ListBucket/DeleteObject` on the bucket). Databricks' *Create metastore*
   wizard generates the exact trust policy.
3. In the **Account console → Catalog → Create metastore**, point it at the bucket + role, then **assign it to your
   workspace**.

> These two resources — the **S3 bucket** and the **IAM role** — are the concrete AWS artifacts to screenshot for your
> own reference; they're what "Unity Catalog on AWS" physically means.

## Step 2 — Enable serverless (or start a small cluster)

- **Serverless** (simplest): Settings → *Compute* → enable serverless for notebooks/jobs. Model Serving and Lakehouse
  Monitoring both require serverless to be available in your region.
- **or a classic cluster**: create a Single-node cluster (e.g. `m5d.large`) on a recent ML-capable runtime. The
  `§5b` cell will then print the real **EC2 instance type**, which is nice proof of EC2 compute.

## Step 3 — Import and run

1. **Workspace → Import → File**, upload `credit_risk_databricks.py` (it imports as a notebook), or clone the GitHub
   repo via **Repos**.
2. Set the widgets at the top:
   - `catalog`: **`main`** (the usual AWS UC default). You can also leave it — the notebook auto-falls-back across
     `main` / `workspace`.
   - `schema`: `credit_risk`, `volume`: `raw`, `deploy_serving`: `yes`.
3. **Run All.** First run downloads the 1.6 GB CSV into a UC Volume (a few minutes), then builds Delta tables, trains
   on the full ~1.35M loans, logs to MLflow, registers the model to UC, deploys serving, and wires monitoring.

## Step 4 — Verify the AWS backing

- The **§5b AWS footprint** cell prints something like:
  ```
  main.credit_risk.loans_features
      -> s3://<your-uc-bucket>/<uuid>/tables/<uuid>
  Compute node type: m5d.large  |  region: us-east-1
  ```
  An `s3://` location = your Lakehouse data is on **S3**. A concrete node type = **EC2**.
- In **Catalog Explorer → loans_features → Lineage**, confirm no leakage column feeds a feature (the firewall proof).
- **AWS console** cross-check (optional, satisfying): open the UC S3 bucket to see the Delta files; open the IAM role
  used by the metastore.

## Cost hygiene (trial)

Serverless + scale-to-zero keeps idle cost ~0, but to be safe when you're done:
- **Delete the Model Serving endpoint** (`credit-risk-pd`) — Serving → Delete. It's scale-to-zero but delete to be sure.
- **Delete / let the cluster auto-terminate.**
- Optionally drop the schema: `DROP SCHEMA main.credit_risk CASCADE;` (also removes the Volume + the 1.6 GB CSV in S3).
- Lakehouse Monitoring runs a small refresh job; delete the monitor if you don't want it recurring.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CREATE SCHEMA` permission denied | You lack UC privileges — grant `CREATE SCHEMA`/`CREATE VOLUME` on the catalog, or use a catalog you own. |
| No `main` catalog | Metastore not attached (Step 1), or create one: `CREATE CATALOG main;` if you have a managed storage root. |
| Model Serving cell errors | Serverless Model Serving not enabled or not available in your region — the notebook skips it gracefully; the rest still runs. |
| Lakehouse Monitoring cell errors | Same as above (serverless required); the SQL drift query still demonstrates the vintage gap. |
| CSV download slow/fails | Re-run the download cell; it's idempotent and resumes by size check. |

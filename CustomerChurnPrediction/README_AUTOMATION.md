# ChurnGuard automated batch prediction — phase 1

Copy `.github/workflows/churnguard-predict.yml` and `scripts/predict_changed.py` into the same paths in your ChurnGuard repository. Keep the notebook-exported `models/feature_schema.json` and its referenced `.joblib` model under `models/`, and ensure `requirements.txt` contains compatible versions of pandas, numpy, scikit-learn, xgboost, joblib and openpyxl.

Any **added or modified** `.csv` or `.xlsx` in `data/` (including nested folders) on `main` triggers the workflow. It scores **only changed files**. Deletions alone do not trigger processing. A manual workflow run processes all supported datasets under `data/`.

The script uses the same validation bounds, category lists, churn thresholds, recommended actions and probability-weighted revenue exposure formula as the uploaded notebook's Streamlit app. It requires `accountid` and all trained features. It writes per-file prediction CSVs and a JSON summary, downloadable from the Actions run artifacts. Files that are not customer feature tables will fail validation and cause a visible failed run; keep unrelated CSV/XLSX outside `data/` or add a schema-based skip policy.

**Phase 2 — persistent alert deduplication:** GitHub runners are ephemeral. Do not reuse the Streamlit app's local SQLite file. Configure an external durable store (for example, PostgreSQL) with a unique key for account ID + risk segment, and configure SMTP credentials as GitHub Actions secrets. Only then enable automated email alerts; this bundle intentionally sends no email, avoiding duplicate or untracked alerts.

**Security:** Do not commit real customer personal data to public GitHub repositories. Prefer a private repository and restricted Actions artifact access. Review artifact retention and data governance requirements.

**Testing:** Run `python scripts/predict_changed.py --all` from the repo root with the trained model present. Then commit a valid sample customer file to `data/` and inspect the Actions run and artifact. Re-running the same commit with `--all` regenerates outputs; alerts are not enabled.

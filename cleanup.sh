#!/bin/bash

echo "############################################"
echo "#       Project Cleanup Utility (Unix)      #"
echo "############################################"

DIR_COUNT=0
FILE_COUNT=0

echo "[1/6] Cleaning cache and checkpoint directories..."
# Find and remove cache-related directories, including subdirectories
find . -type d \( \
    -name "__pycache__" \
    -o -name ".ipynb_checkpoints" \
    -o -name ".pytest_cache" \
    -o -name "*.egg-info" \
    -o -name ".mypy_cache" \
    -o -name ".dvc" \
\) -exec rm -rf {} + && echo "  ✔ Removed cache-related directories"

echo "[2/6] Removing Python cache files (.pyc, .pyo, .pyd)..."
# Find and remove all Python bytecode files, including in subdirectories
PY_CACHE_COUNT=$(find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -exec rm -f {} + | wc -l)
echo "  ✔ Removed $PY_CACHE_COUNT Python bytecode files"

echo "[3/6] Removing temporary and backup files (.tmp, *~, .bak)..."
# Find and remove temporary and backup files in subdirectories
TMP_COUNT=$(find . -type f \( -name "*.tmp" -o -name "*~" -o -name "*.bak" \) -exec rm -f {} + | wc -l)
echo "  ✔ Removed $TMP_COUNT temp/backup files"

echo "[4/6] Removing data outputs and logs (.tsv, .log, .json, .xlsx)..."
# Find and remove data output files and logs
DATA_COUNT=$(find . -type f \( -name "*.tsv" -o -name "*.log" -o -name "*.xlsx" \) -exec rm -f {} + | wc -l)
echo "  ✔ Removed $DATA_COUNT raw data and log files (excluding .csv)"

echo "[5/6] Removing ML model files (.pkl, .npy, .npz, .joblib, .h5)..."
# Find and remove model and data artifacts in subdirectories
MODEL_COUNT=$(find . -type f \( -name "*.pkl" -o -name "*.joblib" -o -name "*.npy" -o -name "*.npz" -o -name "*.h5" -o -name "*.ckpt" \) -exec rm -f {} + | wc -l)
echo "  ✔ Removed $MODEL_COUNT model/data artifacts"

echo "[6/6] Clearing pip cache..."
# Clear pip cache if pip is installed
if command -v pip &> /dev/null; then
    pip cache purge
    echo "  ✔ Pip cache cleared"
else
    echo "  ⚠️ pip not found — skipping cache purge"
fi

echo
echo "✅ Cleanup complete! CSV files have been preserved."

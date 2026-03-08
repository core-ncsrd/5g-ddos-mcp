#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# NCSRD-DS-5GDDoS Dataset Downloader
#
# Dataset: NCSRD-DS-5GDDoS v3.0
# DOI:     https://doi.org/10.5281/zenodo.13900057
# License: CC BY 4.0
# Size:    ~620 MB (7 files)
#
# Usage:
#   chmod +x scripts/download_dataset.sh
#   ./scripts/download_dataset.sh
#
# The script downloads all 7 dataset files to ./data/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ZENODO_RECORD_ID="13900057"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files"

# Dataset files and their Zenodo filenames
declare -A FILES=(
    ["amari_ue_data_merged_with_attack_number.csv"]="amari_ue_data_merged_with_attack_number.csv"
    ["amari_ue_data_classic_tabular.csv"]="amari_ue_data_classic_tabular.csv"
    ["amari_ue_data_mini_tabular.csv"]="amari_ue_data_mini_tabular.csv"
    ["enb_counters_data_classic_tabular.csv"]="enb_counters_data_classic_tabular.csv"
    ["enb_counters_data_mini_tabular.csv"]="enb_counters_data_mini_tabular.csv"
    ["mme_counters.csv"]="mme_counters.csv"
    ["summary_report.xlsx"]="summary_report.xlsx"
)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          NCSRD-DS-5GDDoS Dataset Downloader                 ║"
echo "║  DOI: https://doi.org/10.5281/zenodo.13900057              ║"
echo "║  License: CC BY 4.0                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$DATA_DIR"
echo "Download directory: $DATA_DIR"
echo ""

TOTAL=${#FILES[@]}
COUNT=0

for filename in "${!FILES[@]}"; do
    COUNT=$((COUNT + 1))
    TARGET="$DATA_DIR/$filename"
    URL="${BASE_URL}/${FILES[$filename]}?download=1"

    if [[ -f "$TARGET" ]]; then
        SIZE=$(du -sh "$TARGET" | cut -f1)
        echo "[$COUNT/$TOTAL] SKIP  $filename ($SIZE already exists)"
        continue
    fi

    echo "[$COUNT/$TOTAL] DOWNLOADING  $filename"
    echo "              URL: $URL"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$TARGET" "$URL"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$TARGET" "$URL"
    else
        echo "ERROR: neither wget nor curl found. Install one and retry."
        exit 1
    fi

    SIZE=$(du -sh "$TARGET" | cut -f1)
    echo "              ✅ Saved: $TARGET ($SIZE)"
    echo ""
done

echo ""
echo "✅ Download complete! Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
echo ""
echo "Next step: train the classifier"
echo "  python scripts/train_model.py"

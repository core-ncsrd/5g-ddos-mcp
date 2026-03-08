# NCSRD-DS-5GDDoS Dataset — Download Instructions

This directory is where the dataset files must be placed before training the model.

## Dataset Details

| Field     | Value                                                         |
|-----------|---------------------------------------------------------------|
| Name      | NCSRD-DS-5GDDoS v3.0                                         |
| DOI       | [10.5281/zenodo.13900057](https://doi.org/10.5281/zenodo.13900057) |
| URL       | https://zenodo.org/records/13900057                          |
| License   | Creative Commons Attribution 4.0 (CC BY 4.0)                |
| Total Size| ~620 MB (7 files)                                            |
| Published | October 7, 2024                                              |
| Authors   | NCSRD + Space Hellas (EU Horizon PRIVATEER, Grant 101096110) |

---

## Option 1 — Automatic Download (recommended)

```bash
chmod +x scripts/download_dataset.sh
./scripts/download_dataset.sh
```

This downloads all 7 files from Zenodo into `./data/` automatically.

---

## Option 2 — Manual Download

1. Go to: https://zenodo.org/records/13900057
2. Click **Download all** or download individual files
3. Place the CSV/XLSX files in the `./data/` directory

Expected directory layout after download:
```
data/
├── amari_ue_data_merged_with_attack_number.csv   ← 241.5 MB (primary ML file)
├── amari_ue_data_classic_tabular.csv             ← 143.6 MB
├── amari_ue_data_mini_tabular.csv                ←  87.3 MB
├── enb_counters_data_classic_tabular.csv         ←  72.2 MB
├── enb_counters_data_mini_tabular.csv            ←  38.2 MB
├── mme_counters.csv                              ←  37.1 MB
└── summary_report.xlsx                           ←  18.0 kB
```

---

## File Descriptions

### `amari_ue_data_merged_with_attack_number.csv` ← Start here
The **primary ML-ready file**. Contains merged UE telemetry with the `attack_number` label column:
- `0` = benign (YouTube streaming, Skype VoIP)
- `1` = SYN flooding
- `2` = UDP flooding
- `3` = ICMP flooding
- `4` = DNS flooding
- `5` = GTP-U flooding (5G-specific, critical severity)

Key columns: `ul_bitrate`, `dl_bitrate`, `ul_retx`, `dl_retx`, `ul_mcs`, `dl_mcs`,
`cell_id`, `rnti`, `5g_tmsi`, `imeisv`, `ip`, `attack_number`

### `amari_ue_data_classic_tabular.csv`
Per-UE metrics from Amarisoft Classic cells (Cell 1 & 3). Same schema as merged but unlabeled.

### `amari_ue_data_mini_tabular.csv`
Per-UE metrics from Amarisoft Mini (Cell 2). Same schema as Classic.

### `enb_counters_data_classic_tabular.csv`
Cell-level eNB counters from Classic cells:
`enb_dl_bitrate`, `enb_ul_bitrate`, `enb_dl_usage_ratio`, `enb_ul_usage_ratio`, `gnb_cpu_load`

### `enb_counters_data_mini_tabular.csv`
Same as above for the Mini cell.

### `mme_counters.csv`
NAS-layer MME/AMF counters:
`pdu_session_setup_request`, `pdu_session_setup_response`, `paging_attempts`,
`context_setup_request`, `context_setup_response`, `abnormal_release_count`

### `summary_report.xlsx`
Attack summary: IMEISV, IPs, device types, and attack type per malicious UE.
UE hardware: Huawei P40, Raspberry Pi 4 + 5G Hat, Waveshare Industrial 5G Router,
D-Link DWR-2101, Waveshare 5G CPE Box.

---

## After Downloading

```bash
# Verify files exist
python -c "from src.data.loader import check_dataset_available; import json; print(json.dumps(check_dataset_available(), indent=2))"

# Train the model
python scripts/train_model.py

# Quick test with 100k rows
python scripts/train_model.py --nrows 100000
```

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{ncsrd_5gddos_2024,
  title     = {NCSRD-DS-5GDDoS: 5G Radio & Core Metrics -- DDoS Attack Dataset},
  author    = {NCSRD and Space Hellas},
  year      = {2024},
  doi       = {10.5281/zenodo.13900057},
  publisher = {Zenodo},
  license   = {CC BY 4.0},
  note      = {EU Horizon PRIVATEER project, Grant 101096110}
}
```

---

## Note on the MCP Server

The MCP server can run **without** the dataset in demonstration mode using:
- Rule-based heuristics for `detect_anomaly`
- Reference incidents from the dataset paper for `query_history`
- LLM-only responses for `explain_attack` and `recommend_response`

Training the model (`scripts/train_model.py`) unlocks full ML-powered detection.

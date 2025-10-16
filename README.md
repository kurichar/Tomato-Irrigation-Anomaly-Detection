# Tomato Irrigation - EDA & Isolation Forest Anomaly Detection

**Project aim.** Analyze sensor data from a tomato field experiment with three irrigation regimes (L100 ≈ 100%, L60 ≈ 60%, L30 ≈ 30% of the Irriframe recommendation). We tidy the raw logs into a single **long-format** CSV and detect anomalous soil-humidity behavior with an **Isolation Forest**, optionally explaining anomalies with **SHAP** and grouping them into interpretable periods.

> Context used in the modeling choices:  
> • Early phase: all lines irrigated as I100 until flowering (so early cross-line comparisons are not meaningful).  
> • Two sensor-repositioning events occurred; data are considered reliable **after** the second event.

---

### Required columns in `long_format_data.csv`
- `dt` (ISO timestamp), `line` ∈ {`L100`,`L60`,`L30`}
- Environment: `co2_env`, `humidity_env`, `temperature_env`, `pressure_env`
- Soil probe: `humidity`, `temperature`, `electrical_conductivity`
- Irrigation counter: `current_volume`

> The model notebook assumes these names and types (`dt` → datetime; `line` → categorical).

---

## Quickstart

**Python**: tested on **3.13.3** (works on 3.11+). Create a fresh venv and install:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip wheel
pip install "pandas==2.3.0" "numpy==1.26.4" "scikit-learn==1.7.0" "matplotlib>=3.10.3" "plotly==6.3.0" "shap==0.48.0"
```
If SHAP wheels aren’t available for your platform, you may omit `shap`; notebooks will still run and write the CSVs (explanation fields will be empty).

---

## How to run (order matters)

1. **EDA Notebook.ipynb** → reads raw logs, outputs **`long_format_data.csv`**.  
2. **Isolation Forest Anomaly Detection.ipynb** → reads the long-format CSV and writes:  
   - `pred_points_<LINE>.csv` (per 10‑min step)  
   - `pred_periods_<LINE>.csv` (contiguous spans of anomalies)

Use relative paths only (repo root).

---

## Core modeling logic 

- **Features**: time‑series transforms on soil humidity , EC and soil temperature (lags, rolling means/stdev over short windows, simple slopes) + selected environment vars.  
- **Scaling & fit**: `StandardScaler` fit on **training subset only**; `IsolationForest(n_estimators=800, max_samples=512, max_features=0.8, bootstrap=True, random_state=42)`.  
- **Thresholding**: a timestamp is anomalous if its `decision_function` is in the **lowest q‑quantile** of **training** scores (default `q=0.01`).  
- **Evaluation**: predictions are compared against predeclared **ground‑truth (GT) windows** (early I100 phase; two sensor events). We report **precision** only.

---

## Outputs & interpretation

### `pred_points_<LINE>.csv`
- `score`: Isolation Forest **decision_function** (higher ⇒ more normal; lower/negative ⇒ more anomalous).  
- `is_anom`: boolean anomaly flag after thresholding.  
- `gt_window_label`: GT label if the timestamp lies inside a GT window.  
- `plain_english`, `shap_top1..3`, `shap_summary`: textual reasons (filled only when SHAP is enabled and `is_anom=True`).

### `pred_periods_<LINE>.csv`
- `start`, `end`: boundaries of grouped contiguous anomalies.  
- `n_points`: number of anomalous timestamps in the span.  
- `gt_window_label`: any GT window(s) intersecting the span.  
- `shap_period_summary`: top features by mean |SHAP| inside the period (if SHAP enabled).

### Notebook metrics (printed at the end)
- **Precision (point‑wise)**: fraction of **predicted anomaly points** that fall within GT windows.  
- **Precision (period‑wise)**: fraction of **predicted anomaly periods** intersecting any GT window.

> High precision indicates flagged anomalies largely align with known non‑comparable/“event” windows.

### Interactive Plot
A very interactive plot visualizing anomalies is generated, describing anomalies top contributing features using SHAP.  
<img width="2152" height="1100" alt="Anomaly plot" src="https://github.com/user-attachments/assets/5a5bf05b-0126-4d15-b1a6-038193c5d61c" />
<img width="946" height="339" alt="image" src="https://github.com/user-attachments/assets/c6e8b672-b874-452b-bc63-d9d048d2d4d4" />

For a fully offline interactive version, see [docs/soil_humidity_L30.html](docs/soil_humidity_L30.html).

---

## Acknowledgments / provenance
Experiment background and clarifications are based on the authors’ email responses (private correspondence). The irrigation regimes follow the Italian **Irriframe** guideline; see the associated publication for broader agronomic context.

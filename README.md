# Energy-Aware Cloud Workload Predictor

An AI/ML-based cloud workload forecasting and energy-aware server simulation system developed as an MCA mini-project under Green Computing.

The project uses historical workload data to predict future CPU demand and simulate how intelligent server resource allocation can help reduce estimated energy consumption.

## Project Overview

Cloud data centres consume significant energy to process varying workloads. Keeping all servers active during periods of low demand can lead to unnecessary power consumption.

This project explores how machine learning can forecast upcoming workloads and how a rule-based server management policy can use those predictions to simulate more energy-efficient resource allocation.

The system provides:
- Workload forecasting for 1-hour, 2-hour, 3-hour and 4-hour horizons.
- Random Forest Regression models for CPU workload prediction.
- Comparison with a persistence-based forecasting baseline.
- Energy-aware server allocation simulation.
- Interactive Streamlit dashboard for visualising predictions, evaluation metrics and estimated energy savings.
- Support for both synthetic demonstration data and processed Alibaba Cluster Trace 2018 data.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data processing and numerical operations |
| Scikit-learn | Random Forest Regression and model evaluation |
| Matplotlib / Streamlit charts | Data visualisation |
| Streamlit | Interactive dashboard |
| Joblib | Saving and loading trained ML models |
| Git & GitHub | Version control and project hosting |

## Dataset

The project supports two datasets:

**1. Alibaba Cluster Trace 2018**

The project uses the `machine_usage.csv` workload trace from Alibaba Cluster Trace 2018.

The original dataset is processed into five-minute workload intervals. The processed dataset contains CPU, memory and network-related workload information.

- Processed records: 2,248
- Sampling interval: 5 minutes
- Coverage: Approximately 8 days
- Missing intervals: Short gaps are interpolated; longer gaps are excluded to avoid introducing unreliable values.

The original large raw trace file is not included in this repository. The processed datasets and preprocessing scripts are included.

**2. Synthetic Demonstration Dataset**

A synthetic workload dataset is also included to demonstrate the forecasting and simulation pipeline independently of the Alibaba trace.

## Methodology

### 1. Data Preprocessing
- Process raw workload records.
- Aggregate workload observations into five-minute intervals.
- Handle missing values and gaps.
- Prepare time-based and historical workload features.

### 2. Workload Prediction

Random Forest Regression is used to predict future CPU workload at four different forecasting horizons:

- 1 hour
- 2 hours
- 3 hours
- 4 hours

Historical CPU workload, rolling averages, memory usage, network usage and time-based features are used as model inputs.

### 3. Model Evaluation

The models are evaluated using a chronological training, validation and testing approach.

The following evaluation metrics are calculated:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The Random Forest models are also compared against a persistence baseline, which assumes that the latest observed CPU workload continues into the future.

### 4. Energy-Aware Server Simulation

A rule-based simulation uses predicted workload to estimate the number of active servers required.

It considers:
- Predicted workload and a configurable safety margin.
- Server capacity.
- Active and idle server power assumptions.
- Consecutive low-demand intervals before reducing active servers.
- Unserved demand when simulated capacity is insufficient.

The simulation compares baseline energy consumption, where all configured servers remain active, against the energy estimated under the proposed policy.

## Model Performance

The following results were obtained using the processed Alibaba workload dataset.

| Forecast Horizon | Random Forest MAE | Persistence MAE |
|---|---:|---:|
| 1 Hour | 4.827 | 5.408 |
| 2 Hours | 5.853 | 6.877 |
| 3 Hours | 5.655 | 7.668 |
| 4 Hours | 5.960 | 8.344 |

The Random Forest model achieved lower MAE than the persistence baseline across all four forecasting horizons in this evaluation.

These results represent an initial experiment on the processed trace and should not be interpreted as a guarantee of performance on other workloads.

## Interactive Dashboard

The Streamlit dashboard provides:

- Dataset selection.
- Forecast horizon selection.
- Actual versus predicted workload visualisation.
- Model performance metrics.
- Random Forest and persistence baseline comparison.
- Prediction error analysis.
- Server configuration controls.
- Active server and estimated energy consumption visualisation.
- Unserved demand monitoring.

## Project Structure

```text
Energy-Aware-Cloud-Workload-Predictor/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── workload_5min.csv
│   ├── alibaba_workload_5min.csv
│   └── alibaba_workload_5min_clean.csv
│
├── models/
│   ├── rf_1h.joblib
│   ├── rf_2h.joblib
│   ├── rf_3h.joblib
│   ├── rf_4h.joblib
│   └── alibaba_rf_*.joblib
│
├── outputs/
│   ├── metrics.csv
│   ├── predictions.csv
│   ├── simulation.csv
│   ├── alibaba_metrics.csv
│   └── alibaba_predictions.csv
│
├── src/
│   ├── generate_data.py
│   ├── preprocess_alibaba.py
│   ├── fix_alibaba_gaps.py
│   ├── train.py
│   └── simulate.py
│
└── tests/
    └── test_simulation.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rimjhim-16/Energy-Aware-Cloud-Workload-Predictor.git
cd Energy-Aware-Cloud-Workload-Predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open the localhost URL displayed in the terminal to access the interactive dashboard.

The repository already includes processed datasets, trained models and evaluation outputs, so you can explore the dashboard without retraining the models.

## Important Limitations

- The project is an educational prototype and not a production cloud orchestration system.
- Energy consumption is estimated using assumed server power values (active and idle power), not measured from physical servers.
- The server capacity and allocation policy are configurable simulation assumptions.
- The Alibaba trace covers a limited observation period, so results should be validated on additional datasets before drawing broader conclusions.
- The energy simulation uses the 1-hour prediction stream, even when another forecasting horizon is selected for viewing in the dashboard.
- The simulation reports unserved demand; it does not implement real workload migration or request queuing.

## Future Scope

- Evaluate the models on larger and more diverse cloud workload datasets.
- Explore advanced forecasting models such as LSTM and XGBoost.
- Incorporate dynamic server power models.
- Improve server allocation using optimisation or reinforcement learning.
- Extend the simulation to include workload migration and request queuing.
- Evaluate the approach under more realistic data-centre configurations.

## Author

**Rimjhim Saxena**

MCA (Artificial Intelligence & Data Science)

Graphic Era University

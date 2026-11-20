# Energy-Aware Cloud Workload Predictor
A runnable MCA Green Computing demonstration.

## Important
The included CSV is synthetic demonstration data so the complete pipeline can run immediately. Do not describe results as Alibaba trace results. The Alibaba 2018 machine_usage file has a different schema and requires a separate verified preprocessing step. Energy figures are simulation estimates based on assumed power values, not measured data-centre savings.

## Windows setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate.

## Run
```powershell
python src/generate_data.py
python src/train.py
python src/simulate.py
streamlit run app.py
```
Open the localhost URL printed by Streamlit.

## Method
Five-minute synthetic workload; four direct Random Forest regressors at 1-4 hour horizons; chronological 70/15/15 split; validation 90th percentile absolute error for safety-margin reference; test set used for final metrics only; persistence baseline; MAE/MSE/RMSE.

## Limitations
This is a simplified educational simulation, not a production cloud controller. Power, server capacity and policy parameters are assumptions. Unserved demand is reported rather than queued.

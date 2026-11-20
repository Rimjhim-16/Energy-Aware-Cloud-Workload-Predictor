
import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from src.simulate import simulate

R = Path(__file__).resolve().parent
O = R / "outputs"

st.set_page_config(
    page_title="Energy-Aware Cloud Workload Predictor",
    layout="wide"
)

st.title("⚡ Energy-Aware Cloud Workload Predictor")
st.caption(
    "Green Computing | Machine learning forecasts "
    "+ energy-aware server simulation"
)

# Dataset selection
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Alibaba Real Trace", "Synthetic Demo"]
)

if dataset == "Alibaba Real Trace":
    metrics_file = O / "alibaba_metrics.csv"
    predictions_file = O / "alibaba_predictions.csv"
    dataset_description = (
        "Alibaba Cluster Trace 2018 | "
        "Real machine-level CPU, memory and network measurements. "
        "A sample of 2 million source records was processed."
    )
else:
    metrics_file = O / "metrics.csv"
    predictions_file = O / "predictions.csv"
    dataset_description = (
        "Synthetic demonstration dataset generated for "
        "initial pipeline development."
    )

if not metrics_file.exists() or not predictions_file.exists():
    st.warning("Required dataset files are missing.")
    st.stop()

metrics = pd.read_csv(metrics_file)

p = pd.read_csv(
    predictions_file,
    parse_dates=["timestamp"]
)

st.info(dataset_description)

# Dataset Overview
st.subheader("Dataset Overview")

if dataset == "Alibaba Real Trace":
    data_file = R / "data" / "alibaba_workload_5min_clean.csv"
else:
    data_file = R / "data" / "workload_5min.csv"

if data_file.exists():
    dataset_df = pd.read_csv(
        data_file,
        parse_dates=["timestamp"]
    )

    first_time = dataset_df["timestamp"].min()
    last_time = dataset_df["timestamp"].max()

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Processed Records",
        f"{len(dataset_df):,}"
    )

    d2.metric(
        "Sampling Interval",
        "5 Minutes"
    )

    d3.metric(
        "CPU Forecast Target",
        "CPU Utilisation"
    )

    d4.metric(
        "Resource Features",
        "3"
    )

    st.write(
        f"**Time Coverage:** "
        f"{first_time:%d %b %Y, %H:%M} to "
        f"{last_time:%d %b %Y, %H:%M}"
    )

    st.caption(
        "Resource features: CPU utilisation, memory utilisation "
        "and network traffic."
    )

    if dataset == "Alibaba Real Trace":
        st.caption(
            "Note: Alibaba trace timestamps are relative to the "
            "trace timeline and have been assigned a display-date "
            "anchor. A long gap remains between two continuous "
            "segments in the processed data."
        )
    else:
        st.caption(
            "Note: This is a synthetically generated demonstration "
            "dataset."
        )


# Sidebar controls
h = st.sidebar.selectbox(
    "Forecast horizon (hours)",
    [1, 2, 3, 4]
)

# Simulation settings are applied together when the user clicks the button.
if "simulation_config" not in st.session_state:
    st.session_state.simulation_config = {
        "servers": 10,
        "capacity": 10,
        "margin": 5,
        "low_steps": 6
    }

with st.sidebar.form("simulation_settings"):
    st.markdown("### ⚙️ Simulation Settings")
    servers_input = st.slider(
        "Number of servers", 4, 20,
        st.session_state.simulation_config["servers"]
    )
    cap_input = st.slider(
        "Capacity per server", 5, 20,
        st.session_state.simulation_config["capacity"]
    )
    margin_input = st.slider(
        "Safety margin", 0, 20,
        st.session_state.simulation_config["margin"]
    )
    low_input = st.slider(
        "Low-demand steps", 2, 24,
        st.session_state.simulation_config["low_steps"]
    )
    apply_settings = st.form_submit_button(
        "Apply Simulation Settings",
        type="primary",
        use_container_width=True
    )

if apply_settings:
    st.session_state.simulation_config = {
        "servers": servers_input,
        "capacity": cap_input,
        "margin": margin_input,
        "low_steps": low_input
    }
    st.sidebar.success("Simulation settings applied.")

servers = st.session_state.simulation_config["servers"]
cap = st.session_state.simulation_config["capacity"]
margin = st.session_state.simulation_config["margin"]
low = st.session_state.simulation_config["low_steps"]

# Model metrics
selected_metrics = metrics[
    metrics.horizon_hours == h
]

if selected_metrics.empty:
    st.warning("Metrics unavailable for this horizon.")
    st.stop()

m = selected_metrics.iloc[0]

st.subheader(f"{h}-Hour Forecast Performance")

a, b, c = st.columns(3)

a.metric("MAE", f"{m.MAE:.2f}")
b.metric("RMSE", f"{m.RMSE:.2f}")
c.metric("Persistence MAE", f"{m.Persistence_MAE:.2f}")

# Forecast chart
forecast = p[p.horizon_hours == h].sort_values(
    "timestamp"
).tail(400)

st.subheader(f"{h}-Hour Workload Forecast")

st.plotly_chart(
    px.line(
        forecast,
        x="timestamp",
        y=["actual", "predicted", "persistence"],
        labels={
            "value": "CPU Utilisation (%)",
            "timestamp": "Timestamp",
            "variable": "Series"
        }
    ),
    width="stretch"
)

# Model comparison
st.subheader("Model Evaluation")

st.dataframe(
    metrics.round(3),
    width="stretch"
)


# Prediction Error Analysis
st.subheader("Prediction Error Analysis")

# Calculate prediction errors for the selected horizon
forecast_error = forecast.copy()
forecast_error["prediction_error"] = (
    forecast_error["actual"] - forecast_error["predicted"]
)

col1, col2 = st.columns(2)

with col1:
    st.write("Prediction Error Distribution")

    error_chart = px.histogram(
        forecast_error,
        x="prediction_error",
        nbins=30,
        title="Distribution of Prediction Errors",
        labels={
            "prediction_error": "Prediction Error (CPU %)",
            "count": "Frequency"
        }
    )

    error_chart.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Zero Error"
    )

    st.plotly_chart(error_chart, width="stretch")

with col2:
    st.write("Actual vs Predicted CPU Utilisation")

    comparison_chart = px.scatter(
        forecast_error,
        x="actual",
        y="predicted",
        title="Actual vs Predicted Values",
        labels={
            "actual": "Actual CPU Utilisation (%)",
            "predicted": "Predicted CPU Utilisation (%)"
        }
    )

    st.plotly_chart(comparison_chart, width="stretch")



# Forecast Horizon Performance Comparison
st.subheader("Forecast Horizon Performance Comparison")

comparison_data = metrics[
    ["horizon_hours", "MAE", "Persistence_MAE"]
].rename(
    columns={
        "MAE": "Random Forest",
        "Persistence_MAE": "Persistence Baseline"
    }
).melt(
    id_vars="horizon_hours",
    var_name="Model",
    value_name="Mean Absolute Error"
)

performance_chart = px.bar(
    comparison_data,
    x="horizon_hours",
    y="Mean Absolute Error",
    color="Model",
    barmode="group",
    title="MAE Comparison Across Forecast Horizons",
    labels={
        "horizon_hours": "Forecast Horizon (Hours)",
        "Mean Absolute Error": "MAE (CPU percentage points)"
    }
)

st.plotly_chart(performance_chart, width="stretch")


# Energy simulation
st.subheader("Energy-Aware Server Simulation")

# Simulation uses the selected dataset's 1-hour predictions.
simulation = simulate(
    p,
    servers=servers,
    capacity=cap,
    margin=margin,
    low_steps=low
)

if simulation.empty:
    st.warning("No one-hour predictions available for simulation.")
    st.stop()

baseline = simulation.baseline_kwh.sum()
policy = simulation.policy_kwh.sum()

saving = (
    (1 - policy / baseline) * 100
    if baseline > 0 else 0
)

x, y, z = st.columns(3)

x.metric("Baseline Energy", f"{baseline:.2f} kWh")
y.metric("Policy Energy", f"{policy:.2f} kWh")
z.metric("Estimated Saving", f"{saving:.1f}%")

st.plotly_chart(
    px.line(
        simulation,
        x="timestamp",
        y="active_servers",
        title="Active Server Count"
    ),
    width="stretch"
)

st.plotly_chart(
    px.line(
        simulation,
        x="timestamp",
        y=["baseline_kwh", "policy_kwh"],
        title="Estimated Energy per Interval"
    ),
    width="stretch"
)

st.metric(
    "Unserved Demand Indicator",
    f"{simulation.unserved_demand.sum():.2f}"
)

# Downloadable results
st.subheader("Export Results")
download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        "⬇️ Download Forecast Results",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{h}h.csv",
        mime="text/csv",
        use_container_width=True
    )

with download_col2:
    st.download_button(
        "⬇️ Download Model Metrics",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="model_evaluation_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )

with download_col3:
    st.download_button(
        "⬇️ Download Simulation Results",
        data=simulation.to_csv(index=False).encode("utf-8"),
        file_name="energy_simulation_results.csv",
        mime="text/csv",
        use_container_width=True
    )

st.caption(
    "The forecast and simulation exports contain the results "
    "for the dataset and settings currently selected."
)

# Dataset information and limitations
with st.expander("Dataset, methodology and limitations"):
    st.write(dataset_description)

    st.write(
        "The Random Forest model predicts CPU utilisation "
        "for 1, 2, 3 and 4 hours ahead."
    )

    st.write(
        "Memory and network utilisation are used as historical "
        "input features. They are not separately forecast."
    )

    st.write(
        "Energy values are illustrative estimates based on "
        "assumed active and idle server power consumption. "
        "They are not measured data-centre energy readings."
    )

    st.write(
        "The server allocation policy is simulated locally. "
        "It does not control actual cloud infrastructure."
    )

    st.write(
        "The Alibaba experiment uses a limited sample of "
        "the public trace. A long timestamp gap was excluded "
        "during preprocessing."
    )

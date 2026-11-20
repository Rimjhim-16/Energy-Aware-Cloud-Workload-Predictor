
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "alibaba_workload_5min_clean.csv"
O = ROOT / "outputs"
M = ROOT / "models"

HORIZONS = [1, 2, 3, 4]


def main():
    O.mkdir(exist_ok=True)
    M.mkdir(exist_ok=True)

    d = pd.read_csv(D, parse_dates=["timestamp"])
    d = d.sort_values("timestamp").drop_duplicates("timestamp")
    d = d.set_index("timestamp")

    # Identify continuous five-minute segments.
    d["segment"] = (
        d.index.to_series().diff() != pd.Timedelta(minutes=5)
    ).cumsum()

    x = pd.DataFrame(index=d.index)

    x["hour"] = d.index.hour
    x["weekday"] = d.index.dayofweek

    for lag in [1, 3, 6, 12, 24, 36, 72, 144, 288]:
        x[f"lag_{lag}"] = d.groupby("segment")[
            "cpu_utilization"
        ].shift(lag)

    x["roll_mean_12"] = d.groupby("segment")[
        "cpu_utilization"
    ].transform(
        lambda s: s.shift(1).rolling(12).mean()
    )

    x["memory_lag1"] = d.groupby("segment")[
        "memory_utilization"
    ].shift(1)

    x["network_lag1"] = d.groupby("segment")[
        "network_utilization"
    ].shift(1)

    rows = []
    predlist = []

    for h in HORIZONS:
        z = x.copy()

        z["target"] = d.groupby("segment")[
            "cpu_utilization"
        ].shift(-(h * 12))

        z["current"] = d["cpu_utilization"]
        z["segment"] = d["segment"]

        z = z.dropna()

        cols = list(x.columns)

        # Chronological split within each segment.
        train_parts = []
        val_parts = []
        test_parts = []

        for _, part in z.groupby("segment", sort=False):
            n = len(part)
            a = int(n * 0.70)
            b = int(n * 0.85)

            train_parts.append(part.iloc[:a])
            val_parts.append(part.iloc[a:b])
            test_parts.append(part.iloc[b:])

        tr = pd.concat(train_parts)
        va = pd.concat(val_parts)
        te = pd.concat(test_parts)

        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            print(f"Skipping {h}-hour model: insufficient data")
            continue

        model = RandomForestRegressor(
            n_estimators=70,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=42
        )

        model.fit(tr[cols], tr["target"])

        validation_predictions = model.predict(va[cols])

        margin = float(
            np.quantile(
                np.abs(va["target"] - validation_predictions),
                0.90
            )
        )

        final_model = RandomForestRegressor(
            n_estimators=70,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=42
        )

        fit = pd.concat([tr, va])
        final_model.fit(fit[cols], fit["target"])

        predictions = final_model.predict(te[cols])

        mse = mean_squared_error(te["target"], predictions)

        rows.append({
            "horizon_hours": h,
            "MAE": mean_absolute_error(te["target"], predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "Persistence_MAE": mean_absolute_error(
                te["target"], te["current"]
            ),
            "validation_margin_p90": margin,
            "test_records": len(te)
        })

        prediction_data = pd.DataFrame({
            "timestamp": te.index,
            "actual": te["target"].values,
            "predicted": predictions,
            "persistence": te["current"].values,
            "horizon_hours": h,
            "margin": margin
        })

        predlist.append(prediction_data)

        joblib.dump(
            {
                "model": final_model,
                "features": cols,
                "margin": margin
            },
            M / f"alibaba_rf_{h}h.joblib"
        )

        print(f"Completed {h}-hour model")

    if not rows:
        raise ValueError("Insufficient data to train any model.")

    metrics = pd.DataFrame(rows)

    metrics.to_csv(O / "alibaba_metrics.csv", index=False)

    pd.concat(predlist).to_csv(
        O / "alibaba_predictions.csv",
        index=False
    )

    print("\nAlibaba Training Results:")
    print(metrics.round(3).to_string(index=False))
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

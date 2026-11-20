
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = Path.home() / "Downloads" / "AlibabaTrace" / "machine_usage.csv"
OUTPUT = ROOT / "data" / "alibaba_workload_5min.csv"

COLUMNS = [
    "machine_id", "time_stamp", "cpu_util_percent",
    "mem_util_percent", "mem_gps", "mkpi",
    "net_in", "net_out", "disk_io_percent"
]

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    OUTPUT.parent.mkdir(exist_ok=True)

    totals = None
    total_rows = 0
    max_rows = 2_000_000

    print("Reading Alibaba data in chunks...")

    for chunk in pd.read_csv(
        SOURCE,
        header=None,
        names=COLUMNS,
        usecols=[
            "time_stamp", "cpu_util_percent",
            "mem_util_percent", "net_in", "net_out"
        ],
        chunksize=100_000
    ):
        remaining = max_rows - total_rows
        if remaining <= 0:
            break

        chunk = chunk.iloc[:remaining].copy()
        total_rows += len(chunk)

        numeric_cols = [
            "time_stamp", "cpu_util_percent",
            "mem_util_percent", "net_in", "net_out"
        ]

        for col in numeric_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.dropna(subset=["time_stamp"])

        chunk = chunk[
            chunk["cpu_util_percent"].between(0, 100)
            & chunk["mem_util_percent"].between(0, 100)
        ].copy()

        if chunk.empty:
            continue

        chunk["network_utilization"] = (
            chunk[["net_in", "net_out"]].mean(axis=1)
        )

        chunk["time_bin"] = (
            chunk["time_stamp"] // 300
        ) * 300

        grouped = chunk.groupby("time_bin").agg(
            cpu_sum=("cpu_util_percent", "sum"),
            cpu_count=("cpu_util_percent", "count"),
            memory_sum=("mem_util_percent", "sum"),
            memory_count=("mem_util_percent", "count"),
            network_sum=("network_utilization", "sum"),
            network_count=("network_utilization", "count")
        )

        if totals is None:
            totals = grouped
        else:
            totals = totals.add(grouped, fill_value=0)

        print(f"Read {total_rows:,} source records")

        if total_rows >= max_rows:
            break

    if totals is None or totals.empty:
        raise ValueError("No valid records found.")

    result = pd.DataFrame(index=totals.index)

    result["cpu_utilization"] = (
        totals["cpu_sum"] / totals["cpu_count"]
    )
    result["memory_utilization"] = (
        totals["memory_sum"] / totals["memory_count"]
    )
    result["network_utilization"] = (
        totals["network_sum"] / totals["network_count"].replace(0, np.nan)
    )

    result.index = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(result.index, unit="s")
    )
    result.index.name = "timestamp"

    result = result.reset_index()

    result.to_csv(OUTPUT, index=False)

    print("\nPreprocessing complete!")
    print("Source records processed:", total_rows)
    print("Five-minute intervals created:", len(result))
    print("Time coverage:", result["timestamp"].min(), "to", result["timestamp"].max())
    print("Saved to:", OUTPUT)
    print("\nFirst five rows:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()

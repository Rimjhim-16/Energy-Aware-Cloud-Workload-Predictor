
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "alibaba_workload_5min.csv"
OUTPUT = ROOT / "data" / "alibaba_workload_5min_clean.csv"

def main():
    df = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    complete_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="5min"
    )

    before = len(df)
    df = df.reindex(complete_index)

    missing = df["cpu_utilization"].isna().sum()

    print("Original rows:", before)
    print("Missing intervals:", missing)

    df = df.interpolate(method="time", limit=6)

    df = df.dropna()

    df.index.name = "timestamp"
    df = df.reset_index()

    df.to_csv(OUTPUT, index=False)

    print("\nGap handling complete!")
    print("Clean rows:", len(df))
    print("Remaining missing values:")
    print(df.isna().sum().to_string())
    print("Saved to:", OUTPUT)

if __name__ == "__main__":
    main()

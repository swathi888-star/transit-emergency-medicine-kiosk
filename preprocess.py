import numpy as np
import pandas as pd


def preprocess_data():
  print("Loading raw telemetry log...")
  df = pd.read_csv("raw_vending_telemetry.csv")

  # 1. CLEANING
  # Drop records where Station_ID is missing (from the simulated IoT glitch)
  initial_len = len(df)
  df = df.dropna(subset=["Station_ID"])
  # Drop duplicate records
  df = df.drop_duplicates()
  print(
      f"Cleaned data: Removed {initial_len - len(df)} missing/duplicate rows."
      f" Remaining records: {len(df):,}"
  )

  # Convert Timestamp to datetime
  df["Timestamp"] = pd.to_datetime(df["Timestamp"])
  df["Date"] = df["Timestamp"].dt.date

  # 2. AGGREGATION
  # Resample from individual transactions to Daily Totals per Station and SKU
  print("Aggregating events into daily station-SKU totals...")
  df_daily = (
      df.groupby(["Date", "Station_ID", "Sub_Group", "SKU"])["Quantity_Dispensed"]
      .sum()
      .reset_index(name="Total_Quantity_Sold")
  )

  # Ensure complete grid (all combinations of Date, Station, and SKU) so rolling lags don't break
  df_daily["Date"] = pd.to_datetime(df_daily["Date"])
  all_dates = pd.date_range(
      df_daily["Date"].min(), df_daily["Date"].max(), freq="D"
  )
  stations = df_daily[["Station_ID", "Sub_Group"]].drop_duplicates()
  skus = df_daily["SKU"].drop_duplicates()

  # Create a full Cartesian product frame
  full_grid = pd.MultiIndex.from_product(
      [all_dates, stations["Station_ID"], skus],
      names=["Date", "Station_ID", "SKU"],
  ).to_frame(index=False)

  # Merge back subgroup info
  full_grid = full_grid.merge(stations, on="Station_ID", how="left")

  # Merge with actual sales data, filling missing days with 0 sales
  df_full = full_grid.merge(
      df_daily, on=["Date", "Station_ID", "Sub_Group", "SKU"], how="left"
  )
  df_full["Total_Quantity_Sold"] = df_full["Total_Quantity_Sold"].fillna(0)

  # Sort chronologically for rolling windows
  df_full = df_full.sort_values(by=["Station_ID", "SKU", "Date"]).reset_index(
      drop=True
  )

  # 3. FEATURE ENGINEERING
  print("Engineering time-series lag and rolling features...")

  # Calendar indicators
  df_full["Month"] = df_full["Date"].dt.month
  df_full["DayOfWeek"] = df_full["Date"].dt.weekday
  # pandas weekday(): Mon=0 ... Fri=4, Sat=5, Sun=6 -> weekend is Sat/Sun (>=5)
  df_full["Is_Weekend"] = (df_full["DayOfWeek"] >= 5).astype(int)

  # Re-apply holiday/festival flag mapping for historical alignment
  festival_ranges = [
      pd.date_range(start="2024-04-10", end="2024-04-14"),
      pd.date_range(start="2024-10-09", end="2024-10-13"),
      pd.date_range(start="2024-10-29", end="2024-11-03"),
      pd.date_range(start="2025-03-30", end="2025-04-03"),
      pd.date_range(start="2025-09-28", end="2025-10-02"),
      pd.date_range(start="2025-10-18", end="2025-10-23"),
  ]
  all_festival_dates = set()
  for dr in festival_ranges:
    all_festival_dates.update(dr.date)

  df_full["Is_Festival"] = df_full["Date"].dt.date.isin(
      all_festival_dates
  ).astype(int)

  # Lag and Rolling features grouped by Station and SKU
  group_cols = ["Station_ID", "SKU"]
  df_full["Sales_Lag_1"] = df_full.groupby(group_cols)[
      "Total_Quantity_Sold"
  ].shift(1)
  df_full["Sales_Lag_7"] = df_full.groupby(group_cols)[
      "Total_Quantity_Sold"
  ].shift(7)
  df_full["Sales_Rolling_7"] = df_full.groupby(group_cols)[
      "Total_Quantity_Sold"
  ].transform(lambda s: s.shift(1).rolling(window=7).mean())

  # Fill initial NaN rows created by lags
  df_full = df_full.fillna(0)

  # Save processed training matrix
  output_path = "processed_training_data.csv"
  df_full.to_csv(output_path, index=False)
  print(
      f"Preprocessing complete! Saved feature-engineered matrix with"
      f" {len(df_full):,} rows to {output_path}"
  )


if __name__ == "__main__":
  preprocess_data()
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_demand_model():
  print("Loading preprocessed training matrix...")
  df = pd.read_csv("processed_training_data.csv")
  df["Date"] = pd.to_datetime(df["Date"])

  # Define features and target variable
  # Categorical columns can be encoded or handled directly by LightGBM
  df["Station_ID"] = df["Station_ID"].astype("category")
  df["SKU"] = df["SKU"].astype("category")
  df["Sub_Group"] = df["Sub_Group"].astype("category")

  features = [
      "Station_ID",
      "Sub_Group",
      "SKU",
      "Month",
      "DayOfWeek",
      "Is_Weekend",
      "Is_Festival",
      "Sales_Lag_1",
      "Sales_Lag_7",
      "Sales_Rolling_7",
  ]
  target = "Total_Quantity_Sold"

  # Chronological 80/20 split: hold out the most recent slice of dates as the
  # test set. A random row-wise split shuffles time together, letting the
  # model "test" on days that fall chronologically before some of its
  # training data -- unrealistic for a forecaster that will only ever see
  # the past when predicting the future, and it inflates the reported
  # accuracy versus what the model will actually achieve in production.
  split_date = df["Date"].quantile(0.8, interpolation="nearest")
  train_mask = df["Date"] < split_date
  test_mask = ~train_mask

  X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, target]
  X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, target]

  print(f"Chronological split at {split_date.date()}.")
  print(
      f"Training model on {len(X_train):,} samples (Testing on"
      f" {len(X_test):,} samples)..."
  )

  # Initialize LightGBM Regressor
  model = lgb.LGBMRegressor(
      n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1
  )

  # Train the model
  model.fit(X_train, y_train)

  # Evaluate performance
  preds = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  mae = mean_absolute_error(y_test, preds)

  print("--- Model Evaluation Metrics (Overall) ---")
  print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
  print(f"Mean Absolute Error (MAE):      {mae:.4f}")

  # Per-segment breakdown: a single aggregate RMSE hides a lot when 143
  # station-SKU pairs span wildly different volume (a high-traffic hub vs.
  # a low-volume item at an unrelated station). Slicing by Station and by
  # SKU separately lets the dashboard show where the model is genuinely
  # reliable vs. where a safety-stock heuristic should carry more weight.
  eval_df = df.loc[test_mask, ["Station_ID", "SKU"]].copy()
  eval_df["Actual"] = y_test.values
  eval_df["Predicted"] = preds

  def _segment_metrics(group_col):
    rows = []
    for value, group in eval_df.groupby(group_col, observed=True):
      rows.append({
          "Segment_Type": group_col,
          "Segment_Value": value,
          "RMSE": np.sqrt(mean_squared_error(group["Actual"], group["Predicted"])),
          "MAE": mean_absolute_error(group["Actual"], group["Predicted"]),
          "Test_Samples": len(group),
      })
    return rows

  segment_rows = _segment_metrics("Station_ID") + _segment_metrics("SKU")
  segment_metrics_df = pd.DataFrame(segment_rows)
  segment_metrics_filename = "model_eval_by_segment.csv"
  segment_metrics_df.to_csv(segment_metrics_filename, index=False)
  print(f"Saved per-Station/SKU evaluation breakdown to '{segment_metrics_filename}'.")

  # Save trained model artifact
  model_filename = "demand_model.pkl"
  joblib.dump(model, model_filename)
  print(f"Trained model successfully saved to '{model_filename}'!")

  # Also persist the feature order and the exact categorical levels seen
  # during training. LightGBM encodes pandas "category" columns using each
  # column's integer category codes. If a later inference step (e.g. the
  # Streamlit dashboard) rebuilds these columns via .astype("category") on
  # a smaller DataFrame (say, one station), pandas will assign new codes
  # based on whatever categories happen to be present there -- not the
  # codes the model was trained on. Same station name, silently different
  # code, wrong prediction, and no error raised. Saving the training-time
  # categories lets inference code re-apply them exactly:
  #   df[col] = pd.Categorical(df[col], categories=meta["categories"][col])
  metadata = {
      "features": features,
      "categories": {
          col: df[col].cat.categories.tolist()
          for col in ["Station_ID", "SKU", "Sub_Group"]
      },
      # Saved so any downstream evaluation code (e.g. KioskForecastEngine)
      # can filter to Date >= split_date and report genuinely held-out
      # metrics, instead of accidentally evaluating on rows the model was
      # trained on and reporting an inflated, in-sample accuracy number.
      "split_date": split_date.isoformat(),
  }
  meta_filename = "demand_model_meta.pkl"
  joblib.dump(metadata, meta_filename)
  print(f"Saved feature/category metadata to '{meta_filename}'.")


if __name__ == "__main__":
  train_demand_model()
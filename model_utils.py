from datetime import timedelta
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class KioskForecastEngine:

  def __init__(
      self,
      model_path="demand_model.pkl",
      data_path="processed_training_data.csv",
      meta_path="demand_model_meta.pkl",
  ):
    print("Loading model and training data context...")
    self.model = joblib.load(model_path)
    self.df = pd.read_csv(data_path)
    self.df["Date"] = pd.to_datetime(self.df["Date"])

    # split_date marks where training data ended and held-out test data
    # began (see train_model.py). Used by get_per_group_evaluation() to
    # report genuinely out-of-sample metrics instead of evaluating on rows
    # the model was trained on, which would report an inflated accuracy.
    try:
      meta = joblib.load(meta_path)
      self.split_date = pd.Timestamp(meta["split_date"])
    except (FileNotFoundError, KeyError):
      self.split_date = None
      print(
          f"Warning: could not load split_date from '{meta_path}'."
          " get_per_group_evaluation() will fall back to evaluating on the"
          " full dataset, which includes training rows and will report an"
          " optimistic, in-sample accuracy rather than genuine held-out"
          " performance."
      )

    # These three columns have to be category dtype (not plain strings) both
    # because the model was trained on them as categoricals and because
    # predict_recursive_7_day below reads their categories via .cat.categories
    # to build single-row prediction frames that match the training encoding.
    for col in ["Station_ID", "Sub_Group", "SKU"]:
      self.df[col] = self.df[col].astype("category")

  def get_per_group_evaluation(self):
    """(b) Computes per-Station and per-SKU RMSE/MAE breakdown

    to showcase model behavior across high vs low volume items.

    Restricted to rows on/after split_date (the held-out test period from
    train_model.py) when available. Without this filter, evaluating on
    self.df directly mixes in rows the model was trained on, which
    understates real error and would report a better-looking accuracy than
    the model will actually achieve on unseen data.
    """
    if self.split_date is not None:
      eval_source = self.df[self.df["Date"] >= self.split_date].copy()
    else:
      eval_source = self.df.copy()

    # Run test predictions across the (held-out, when available) frame
    X_eval = eval_source[
        [
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
    ].copy()

    preds = self.model.predict(X_eval)
    eval_df = eval_source.copy()
    eval_df["Prediction"] = preds

    # Group by Station
    station_eval = []
    for station, group in eval_df.groupby("Station_ID", observed=True):
      rmse = np.sqrt(
          mean_squared_error(
              group["Total_Quantity_Sold"], group["Prediction"]
          )
      )
      mae = mean_absolute_error(
          group["Total_Quantity_Sold"], group["Prediction"]
      )
      mean_vol = group["Total_Quantity_Sold"].mean()
      station_eval.append({
          "Station_ID": station,
          "Mean_Daily_Sales": round(mean_vol, 2),
          "RMSE": round(rmse, 2),
          "MAE": round(mae, 2),
      })

    # Group by SKU
    sku_eval = []
    for sku, group in eval_df.groupby("SKU", observed=True):
      rmse = np.sqrt(
          mean_squared_error(
              group["Total_Quantity_Sold"], group["Prediction"]
          )
      )
      mae = mean_absolute_error(
          group["Total_Quantity_Sold"], group["Prediction"]
      )
      mean_vol = group["Total_Quantity_Sold"].mean()
      sku_eval.append({
          "SKU": sku,
          "Mean_Daily_Sales": round(mean_vol, 2),
          "RMSE": round(rmse, 2),
          "MAE": round(mae, 2),
      })

    return pd.DataFrame(station_eval), pd.DataFrame(sku_eval)

  def predict_recursive_7_day(self, station_id, sku):
    """(a) 7-day recursive forecast wrapper.

    Predicts Day 1, feeds prediction back as Sales_Lag_1 for Day 2, etc.
    """
    # Get the latest known date and history for this station-sku
    sub_df = self.df[
        (self.df["Station_ID"] == station_id) & (self.df["SKU"] == sku)
    ].sort_values("Date")
    if sub_df.empty:
      return []

    latest_row = sub_df.iloc[-1]
    current_date = latest_row["Date"]

    # Grab recent sales buffer for recursive lag updates
    recent_sales = list(sub_df["Total_Quantity_Sold"].tail(7))

    forecasts = []
    # We will simulate 7 future days
    for day_offset in range(1, 8):
      future_date = current_date + timedelta(days=day_offset)

      # Extract features for prediction
      month = future_date.month
      day_of_week = future_date.weekday()
      # Must match preprocess.py's Is_Weekend definition exactly (Sat/Sun,
      # i.e. >= 5), since that's what the model was actually trained on.
      # A >= 4 cutoff here would flag Friday as a weekend at inference time
      # despite the model never seeing real Fridays labeled that way during
      # training, silently skewing every Friday forecast.
      is_weekend = 1 if day_of_week >= 5 else 0
      is_festival = 0  # Can lookup from festival list if needed

      # Lags derived from our recursive buffer
      sales_lag_1 = recent_sales[-1]
      # preprocess.py fills missing lag values with 0 (df_full.fillna(0)),
      # not the oldest available value -- match that convention here so a
      # short history is treated the same way at inference as at training.
      sales_lag_7 = recent_sales[-7] if len(recent_sales) >= 7 else 0.0
      sales_rolling_7 = np.mean(recent_sales[-7:])

      # Build the single-row frame with plain scalar values first, then cast
      # the categorical columns using self.df's *trained* categories. Putting
      # pd.Categorical(...) objects directly as dict values (the previous
      # approach) makes pandas store the whole Categorical as one nested
      # object per cell instead of a proper category column, so the model
      # never actually sees a valid Station_ID/Sub_Group/SKU value.
      features = pd.DataFrame([{
          "Station_ID": station_id,
          "Sub_Group": latest_row["Sub_Group"],
          "SKU": sku,
          "Month": month,
          "DayOfWeek": day_of_week,
          "Is_Weekend": is_weekend,
          "Is_Festival": is_festival,
          "Sales_Lag_1": sales_lag_1,
          "Sales_Lag_7": sales_lag_7,
          "Sales_Rolling_7": sales_rolling_7,
      }])

      for col in ["Station_ID", "Sub_Group", "SKU"]:
        features[col] = pd.Categorical(
            features[col], categories=self.df[col].cat.categories
        )

      pred = float(self.model.predict(features)[0])
      pred = max(0.0, round(pred, 2))  # Ensure non-negative demand

      forecasts.append({
          "Day": f"Day +{day_offset}",
          "Date": future_date.strftime("%Y-%m-%d"),
          "Confidence": (
              "High (High-Confidence)" if day_offset <= 2 else "Directional Outlook"
          ),
          "Predicted_Demand": pred,
      })

      # Append prediction to buffer for next recursive step
      recent_sales.append(pred)

    return forecasts

  def compute_stock_risk_panel(
      self, current_inventory_dict, station_id, low_stock_threshold=15
  ):
    """(c) Computes Days_of_Stock_Remaining = current_stock / predicted_next_day_demand

    Layered alongside the static rule-based threshold alert.
    low_stock_threshold should match whatever the app's own alert banner
    uses (15/25 in streamlit_app.py) -- a hardcoded different number here
    would show a second, inconsistent "critical" line next to the app's
    real alert for the same SKU.
    """
    risk_rows = []
    # Get next-day forecast for all SKUs at this station
    skus = self.df["SKU"].unique()

    for sku in skus:
      # Get 1-day prediction
      forecasts = self.predict_recursive_7_day(station_id, sku)
      next_day_demand = forecasts[0]["Predicted_Demand"] if forecasts else 1.0
      next_day_demand = max(
          0.1, next_day_demand
      )  # prevent division by zero

      current_stock = current_inventory_dict.get(sku, 15)  # default mock stock

      # Days of stock remaining ratio
      days_remaining = round(current_stock / next_day_demand, 1)

      # Dual alerting logic:
      # 1. Static rule-based threshold check
      static_alert = (
          f"CRITICAL (Stock < {low_stock_threshold})"
          if current_stock < low_stock_threshold
          else "OK"
      )

      # 2. Demand-aware ratio check (< 3 predicted days remaining)
      if days_remaining < 2.0:
        ml_risk_status = "🔴 HIGH RISK (Stockout Imminent)"
      elif days_remaining < 4.0:
        ml_risk_status = "🟡 MODERATE RISK (Reorder Soon)"
      else:
        ml_risk_status = "🟢 HEALTHY"

      risk_rows.append({
          "SKU": sku,
          "Current_Stock": current_stock,
          "Next_Day_Demand_Pred": next_day_demand,
          "Days_Remaining": days_remaining,
          "Static_Threshold_Status": static_alert,
          "ML_Demand_Risk_Status": ml_risk_status,
      })

    return pd.DataFrame(risk_rows)
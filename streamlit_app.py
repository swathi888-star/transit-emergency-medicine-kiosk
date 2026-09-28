import csv
import math
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import KioskForecastEngine

st.set_page_config(
    page_title="Transit Emergency Medicine Kiosk",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Threshold below which a SKU at a given kiosk is considered "low stock".
# Region-specific meds (e.g. Motion Sickness tablets at a Hill Station hub)
# get an earlier, higher-buffer warning than universal "All" items, since
# running out of the medicine a hub exists to dispense is a bigger deal
# than running low on a generic painkiller.
LOW_STOCK_THRESHOLD_UNIVERSAL = 15
LOW_STOCK_THRESHOLD_REGIONAL = 25

# Words too generic to count as a real region match on their own.
_REGION_STOPWORDS = {"all", "and", "the", "for"}

# Generic transit-infrastructure words that show up in almost every
# Location_Name (e.g. "Goa Railway Station", "Udaipur City Station").
# Excluded only from Location_Name keywords, so a descriptive tag word
# like "Stations" (from "Hill Stations") can't false-match every hub
# whose name happens to contain the word "Station".
_LOCATION_GENERIC_WORDS = {
    "airport", "station", "stations", "stand", "central",
    "railway", "port", "city", "junction", "road",
}


def _region_keywords(text, exclude_generic=False):
  """Lowercase, length>=4 keywords from a tag/Sub_Group/Location_Name.

  Length filter drops noise words ('or', 'in'); the stopword set drops
  common connector words. exclude_generic additionally strips transit
  infrastructure words -- only meant for Location_Name, never Sub_Group
  or the catalog tag itself, where those words carry real meaning
  (e.g. "Hill Stations" as a Sub_Group).
  """
  words = re.findall(r"[a-zA-Z]+", str(text).lower())
  keywords = {w for w in words if len(w) >= 4 and w not in _REGION_STOPWORDS}
  if exclude_generic:
    keywords -= _LOCATION_GENERIC_WORDS
  return keywords


def is_region_specific_match(target_tag, sub_group, location_name):
  """True if a catalog Target_Destination_Tag is specific to this hub.

  Replaces the old exact-string check (tag == sub_group), which silently
  failed on compound tags like "Coastal Hubs & Hill Stations" or "Arid
  Deserts & Hyderabad" -- neither equals any Sub_Group value verbatim, so
  those SKUs were never recognized as region-relevant anywhere in the app.
  This does a keyword-overlap match instead, checked against both the
  hub's Sub_Group and its Location_Name (covers tags like "Delhi" or
  "Bengaluru" that name a location directly rather than a Sub_Group).
  "All" is universal by definition, never region-specific.
  """
  if str(target_tag).strip().lower() == "all":
    return False
  tag_words = _region_keywords(target_tag)
  hub_words = _region_keywords(sub_group) | _region_keywords(
      location_name, exclude_generic=True
  )
  return any(tw in hw or hw in tw for tw in tag_words for hw in hub_words)


def get_region_critical_skus(hub_name, _catalog_df, _locations_df):
  """SKUs whose Target_Destination_Tag is specific to this hub (not 'All')."""
  loc_row = _locations_df[_locations_df["Location_Name"] == hub_name].iloc[0]
  sub_group = loc_row["Sub_Group"]
  return {
      row["SKU"]
      for _, row in _catalog_df.iterrows()
      if is_region_specific_match(row["Target_Destination_Tag"], sub_group, hub_name)
  }


@st.cache_data
def get_burn_rate_table(_sales_df):
  """Average daily units sold per (station, SKU), from historical sales.

  Deliberately NOT the ML model: this powers "days remaining" and a smart
  restock suggestion inside the Alerts tab, which needs to work even
  before a model has been trained, and shouldn't add ~150 model.predict
  calls to a tab that's meant to be the fast, always-on rule-based view
  (the model-backed forecast lives in the ML Insights tab instead).
  """
  daily = (
      _sales_df.groupby(["Kiosk_Location", "SKU", "Date"])["Units_Sold"]
      .sum()
      .reset_index()
  )
  avg = (
      daily.groupby(["Kiosk_Location", "SKU"])["Units_Sold"]
      .mean()
      .reset_index()
      .rename(columns={"Units_Sold": "Avg_Daily_Units"})
  )
  return avg.set_index(["Kiosk_Location", "SKU"])["Avg_Daily_Units"].to_dict()


def get_burn_rate(burn_rate_table, hub, sku, floor=0.5):
  """Avg daily units for one station-SKU, floored so division stays sane
  for items with little/no sales history rather than divide-by-zero."""
  return max(floor, burn_rate_table.get((hub, sku), floor))


# ==================== AUDIT LOG (DATA FLYWHEEL) ====================
# Every symptom selection, sale, and restock is captured here, timestamped,
# with station ID and ambient climate risk factor -- this is the raw feed
# the ML Insights tab below trains on. It's the same shared-across-sessions
# pattern as `inventory` (st.cache_resource, not st.session_state), plus a
# CSV mirror on disk so the log survives a server restart and can be handed
# to an offline training job, not just read back within this process.
AUDIT_LOG_PATH = "audit_log.csv"
AUDIT_LOG_COLUMNS = [
    "timestamp", "event_type", "station_id", "sub_group", "primary_risk",
    "symptom_category", "sku", "quantity", "unit_price", "total_price",
    "region_critical", "stock_before", "stock_after", "actor",
]


@st.cache_resource
def get_audit_log():
  if os.path.exists(AUDIT_LOG_PATH):
    try:
      records = pd.read_csv(AUDIT_LOG_PATH).to_dict("records")
    except pd.errors.EmptyDataError:
      return []
    # Earlier builds let a "restock" set stock *below* its current level and
    # logged the drop as a negative quantity. Those rows are stock
    # reductions, not restocks, so relabel them rather than show them as
    # negative restocks (history is kept, just correctly labelled).
    for r in records:
      if r.get("event_type") == "restock" and pd.notna(r.get("quantity")) and r["quantity"] < 0:
        r["event_type"] = "adjustment"
    return records
  return []


def log_event(event_type, **fields):
  """Append one structured event to the shared audit log (memory + disk).

  event_type: "symptom_selected" | "purchase" | "restock" | "adjustment" (legacy)
  """
  record = {col: fields.get(col, "") for col in AUDIT_LOG_COLUMNS}
  record["timestamp"] = datetime.now().isoformat(timespec="seconds")
  record["event_type"] = event_type
  audit_log.append(record)
  file_exists = os.path.exists(AUDIT_LOG_PATH)
  with open(AUDIT_LOG_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=AUDIT_LOG_COLUMNS)
    if not file_exists:
      writer.writeheader()
    writer.writerow(record)


# ==================== ML INSIGHTS ====================
# All three intentionally use plain pandas/numpy (rolling averages,
# linear-trend fits, z-scores) rather than a heavier forecasting library --
# this is enough signal at the current data volume, and each can be swapped
# for Prophet/ARIMA/scikit-learn later without changing what calls them.


def forecast_demand(sales_df, station, sku, horizon_days=7):
  """Linear-trend forecast on daily Units_Sold history for one SKU/station.

  Smooths day-to-day noise with a 7-day rolling average, fits a straight
  line to that, and projects it forward. Returns None if there's too
  little history to fit anything meaningful.
  """
  hist = sales_df[
      (sales_df["Kiosk_Location"] == station) & (sales_df["SKU"] == sku)
  ].copy()
  if len(hist) < 5:
    return None
  hist["Date"] = pd.to_datetime(hist["Date"])
  hist = hist.sort_values("Date")
  hist["day_num"] = (hist["Date"] - hist["Date"].min()).dt.days
  hist["Rolling_Avg"] = hist["Units_Sold"].rolling(7, min_periods=1).mean()
  slope, intercept = np.polyfit(hist["day_num"], hist["Rolling_Avg"], 1)
  last_day = hist["day_num"].max()
  last_date = hist["Date"].max()
  future_day_nums = np.arange(last_day + 1, last_day + 1 + horizon_days)
  future_dates = [
      last_date + timedelta(days=int(d - last_day)) for d in future_day_nums
  ]
  predicted = np.clip(slope * future_day_nums + intercept, 0, None)
  future = pd.DataFrame({"Date": future_dates, "Predicted_Units": predicted})
  return hist, future


def seasonal_category_insights(sales_df, catalog_df):
  """Avg daily units per Symptom_Category per month vs its yearly average.

  Surfaces climate/seasonal spikes -- e.g. Respiratory & Smog Care rising
  in Dec/Jan -- as a ratio so a manager can see how far above (or below)
  a category's normal baseline a given month runs. historical_sales.csv
  already carries Symptom_Category per row (see generate_data.py), so no
  catalog join is needed here.
  """
  merged = sales_df.copy()
  merged["Date"] = pd.to_datetime(merged["Date"])
  merged["Month"] = merged["Date"].dt.month_name()
  monthly = (
      merged.groupby(["Symptom_Category", "Month"])["Units_Sold"]
      .mean()
      .reset_index()
  )
  overall = merged.groupby("Symptom_Category")["Units_Sold"].mean()
  monthly["Overall_Avg"] = monthly["Symptom_Category"].map(overall)
  monthly["Spike_Ratio"] = monthly["Units_Sold"] / monthly["Overall_Avg"]
  return monthly


def detect_anomalies(sales_df, purchases_df, z_thresh=2.0):
  """Flag live purchase events whose quantity is an outlier against that
  SKU/station's historical daily-sales distribution (z-score based).
  """
  baseline = (
      sales_df.groupby(["Kiosk_Location", "SKU"])["Units_Sold"]
      .agg(mean="mean", std="std")
      .reset_index()
  )
  merged = purchases_df.merge(
      baseline,
      left_on=["station_id", "sku"],
      right_on=["Kiosk_Location", "SKU"],
      how="left",
  )
  merged["std"] = merged["std"].replace(0, np.nan)
  merged["z_score"] = (
      pd.to_numeric(merged["quantity"], errors="coerce") - merged["mean"]
  ) / merged["std"]
  return merged[merged["z_score"].abs() >= z_thresh].dropna(subset=["z_score"])

def render_alerts_and_restock_tab(hubs_to_check, restock_hub_choices, actor_label):
  """Low-stock capsule grid + restock control, shared by both the
  single-station Operator view and the full-network Manager view -- the
  only difference between the two is how many hubs are passed in."""
  # Confirmation for a restock that just happened (set by the Confirm
  # Restock handler below, right before it reruns). pop() so it appears
  # exactly once and doesn't linger on later interactions.
  flash = st.session_state.pop("restock_flash", None)
  if flash:
    if flash["alert_cleared"]:
      status_line = "Low-stock alert cleared."
    else:
      status_line = (
          f"Still below the {flash['threshold']}-unit alert threshold, so this"
          " item remains flagged."
      )
    st.markdown(
        f"""
        <div style="background-color: #E8F5E9; border: 2px solid #2E7D32;
             border-left: 8px solid #2E7D32; border-radius: 10px;
             padding: 14px 18px; margin-bottom: 16px;">
          <span style="color: #1B5E20; font-size: 18px; font-weight: 800;">
            ✓ Restocked successfully
          </span><br>
          <span style="color: #1a1a1a; font-size: 15px;">
            {flash['sku']} at {flash['hub']}: {flash['before']} → <b>{flash['after']}</b> units.
            {status_line}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.toast(f"Restocked {flash['sku']} at {flash['hub']} → {flash['after']} units", icon="✅")

  # Per-hub summary: how many SKUs are below threshold, and how many of
  # those are region-critical (alert earlier, since they're the reason
  # travelers use that specific hub).
  hub_summary = {}
  for hub in hubs_to_check:
    region_critical_skus = get_region_critical_skus(hub, catalog_df, locations_df)
    low_items = []
    for sku, qty in inventory[hub].items():
      is_critical = sku in region_critical_skus
      threshold = (
          LOW_STOCK_THRESHOLD_REGIONAL if is_critical else LOW_STOCK_THRESHOLD_UNIVERSAL
      )
      if qty < threshold:
        daily_burn = get_burn_rate(burn_rate_table, hub, sku)
        low_items.append({
            "SKU": sku, "Quantity": qty, "Threshold": threshold,
            "Region_Critical": is_critical,
            "Days_Remaining": round(qty / daily_burn, 1),
            "Daily_Burn": daily_burn,
        })
    low_items.sort(key=lambda a: (not a["Region_Critical"], a["Days_Remaining"]))
    hub_summary[hub] = low_items

  total_alerts = sum(len(v) for v in hub_summary.values())
  hubs_with_alerts = sum(1 for v in hub_summary.values() if v)

  # Higher-contrast banner than the default st.warning box, so the
  # top-line count is the first thing that pulls the eye on this tab.
  if total_alerts:
    st.markdown(
        f"""
        <div style="background-color: #D32F2F; color: #FFFFFF; padding: 16px 20px;
             border-radius: 10px; border: 2px solid #FFFFFF; margin-bottom: 16px;
             box-shadow: 0 4px 10px rgba(0,0,0,0.4);">
          <span style="font-size: 20px; font-weight: 800;">
            ⚠ {total_alerts} item(s) low across {hubs_with_alerts} hub(s)
          </span><br>
          <span style="font-size: 14px;">Tap a hub below. 🔥 = region-critical, alerts sooner.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
  else:
    st.success("All hubs fully stocked.")

  # Capsule grid: one tile per hub, red badge if it has anything low.
  # Operator only ever passes one hub, so skip the grid and go straight
  # to that hub's detail.
  if len(hubs_to_check) > 1:
    st.markdown("**Select a hub:**")
    if "selected_alert_hub" not in st.session_state or st.session_state.selected_alert_hub not in hubs_to_check:
      st.session_state.selected_alert_hub = hubs_to_check[0]

    cols = st.columns(4)
    for idx, hub in enumerate(hubs_to_check):
      n_low = len(hub_summary[hub])
      # Icon-only badge on the capsule button itself; the exact count is
      # shown just below once a hub is selected, so the button stays a
      # quick at-a-glance scan rather than a wall of numbers.
      badge = "⚠️" if n_low else "✅"
      with cols[idx % 4]:
        if st.button(f"{hub}\n{badge}", key=f"hub_capsule_{hub}"):
          st.session_state.selected_alert_hub = hub
          st.rerun()
    selected_hub = st.session_state.selected_alert_hub
    st.markdown("---")
  else:
    selected_hub = hubs_to_check[0]

  # Detail for the selected hub only, not a wall of text for all of them.
  hub_low_items = hub_summary[selected_hub]
  st.markdown(f"### {selected_hub}")
  if hub_low_items:
    st.caption(f"{len(hub_low_items)} item(s) below threshold at this hub.")
    for item in hub_low_items:
      severity = "critical" if item["Days_Remaining"] < 3 else "warning"
      pill_class = "status-pill-critical" if severity == "critical" else "status-pill-warning"
      pill_label = "CRITICAL" if severity == "critical" else "WARNING"
      tag = "🔥 Region-Critical · " if item["Region_Critical"] else ""
      st.markdown(
          f"""
          <div class="alert-item-card {severity}">
            <div>
              <span class="alert-item-label">{tag}{item['SKU']}</span>
              <div class="alert-item-meta">{item['Quantity']} units left ·
                ~{item['Daily_Burn']:.1f}/day burn rate</div>
            </div>
            <div style="text-align: right;">
              <span class="status-pill {pill_class}">{pill_label}</span>
              <div class="alert-item-meta">Est. stockout in {item['Days_Remaining']} days</div>
            </div>
          </div>
          """,
          unsafe_allow_html=True,
      )
  else:
    st.success("Fully stocked.")

  st.subheader("Restock")

  if len(restock_hub_choices) > 1:
    restock_hub = selected_hub
  else:
    restock_hub = restock_hub_choices[0]

  restock_region_critical_skus = get_region_critical_skus(
      restock_hub, catalog_df, locations_df
  )
  sku_options = sorted(
      catalog_df["SKU"].tolist(),
      key=lambda s: s not in restock_region_critical_skus,
  )
  sku_to_update = st.selectbox(
      "SKU / Product",
      sku_options,
      format_func=lambda s: (
          f"🔥 {s} (Region-Critical)" if s in restock_region_critical_skus else s
      ),
      # Keyed per-hub: a bare static key kept the widget's *index* pinned
      # across hub switches (Streamlit remembers selection by key, not by
      # value), so the box looked "frozen" on whatever SKU was selected
      # when you were last on a different hub, even though the options
      # list itself was correct.
      key=f"restock_sku_select_{restock_hub}",
  )
  current_qty = inventory[restock_hub][sku_to_update]
  product_row = catalog_df[catalog_df["SKU"] == sku_to_update].iloc[0]
  product_desc = get_field(
      product_row,
      ["Active_Ingredient_Description", "Product_Name", "Description"],
      default=sku_to_update,
  )

  st.markdown(f"**{product_desc}** — current stock: {current_qty} units")

  # Smart default: top up to roughly a 14-day supply at this SKU/station's
  # historical average daily sales, instead of a flat "+50" guess. Still
  # fully editable -- this is a starting suggestion, not a locked value.
  RESTOCK_TARGET_DAYS = 14
  daily_burn = get_burn_rate(burn_rate_table, restock_hub, sku_to_update)
  suggested_qty = current_qty + math.ceil(daily_burn * RESTOCK_TARGET_DAYS)
  st.caption(
      f"Suggested: {suggested_qty} units (~{RESTOCK_TARGET_DAYS}-day supply at"
      f" {daily_burn:.1f}/day historical average). Adjust as needed."
  )

  # A restock can only add stock, so the floor is the current level. Without
  # this, typing a lower number logged a negative "restock" quantity in the
  # audit trail. current_qty is part of the key so the box resets whenever
  # stock changes (a sale, or another session restocking), instead of
  # holding a stale value that could sit below the new floor.
  new_restock_qty = st.number_input(
      "New quantity after restock",
      min_value=current_qty,
      max_value=max(500, current_qty),
      value=min(max(suggested_qty, current_qty), max(500, current_qty)),
      key=f"restock_input_{restock_hub}_{sku_to_update}_{current_qty}",
  )

  if st.button("Confirm Restock", type="primary"):
    restock_loc_row = locations_df[
        locations_df["Location_Name"] == restock_hub
    ].iloc[0]
    stock_before = inventory[restock_hub][sku_to_update]
    inventory[restock_hub][sku_to_update] = new_restock_qty
    log_event(
        "restock",
        station_id=restock_hub,
        sub_group=restock_loc_row["Sub_Group"],
        primary_risk=restock_loc_row["Primary_Risk"],
        sku=sku_to_update,
        quantity=max(0, new_restock_qty - stock_before),  # units added, never negative
        stock_before=stock_before,
        stock_after=new_restock_qty,
        region_critical=sku_to_update in restock_region_critical_skus,
        actor=actor_label,
    )
    # Don't call st.success() here: st.rerun() restarts the script on the
    # next line, so the message would be wiped before the browser draws
    # it. Stash it in session_state and show it after the rerun instead.
    restocked_threshold = (
        LOW_STOCK_THRESHOLD_REGIONAL
        if sku_to_update in restock_region_critical_skus
        else LOW_STOCK_THRESHOLD_UNIVERSAL
    )
    st.session_state.restock_flash = {
        "sku": sku_to_update,
        "hub": restock_hub,
        "before": stock_before,
        "after": new_restock_qty,
        "alert_cleared": new_restock_qty >= restocked_threshold,
        "threshold": restocked_threshold,
    }
    st.rerun()


def render_audit_log_tab():
  st.subheader("Full Audit Trail")
  st.caption("Sales and restocks are transactions. Symptom clicks are just browsing, shown separately.")
  log_df = pd.DataFrame(audit_log, columns=AUDIT_LOG_COLUMNS)
  if log_df.empty:
    st.info("No events logged yet.")
  else:
    event_types = sorted(log_df["event_type"].dropna().unique())
    # Default to real transactions only -- a symptom click is a customer
    # browsing an option, not a sale or restock, and showing it mixed in
    # by default reads as a phantom transaction.
    transaction_types = [t for t in event_types if t in ("purchase", "restock")]
    event_filter = st.multiselect(
        "Show:", options=event_types, default=transaction_types or event_types
    )
    display_df = log_df[log_df["event_type"].isin(event_filter)].sort_values(
        "timestamp", ascending=False
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.Column(width="medium"),
            "event_type": st.column_config.Column(width="small"),
            "station_id": st.column_config.Column(width="medium"),
            "sku": st.column_config.Column(width="small"),
            "quantity": st.column_config.Column(width="small"),
            "unit_price": st.column_config.Column(width="small"),
            "total_price": st.column_config.Column(width="small"),
            "region_critical": st.column_config.Column(width="small"),
            "stock_before": st.column_config.Column(width="small"),
            "stock_after": st.column_config.Column(width="small"),
        },
    )
    st.download_button(
        "⬇️ Download Full Audit Log (CSV)",
        data=log_df.to_csv(index=False),
        file_name="audit_log.csv",
        mime="text/csv",
    )


def render_ml_insights_tab():
  st.subheader("ML-Powered Stock Intelligence")

  tab_risk, tab_detail, tab_eval, tab_seasonal, tab_anomaly = st.tabs(
      ["Network Risk", "Kiosk Detail", "Model Evaluation", "Seasonal Trends", "Anomalies"]
  )

  model_unavailable_msg = (
      "Demand model not found — run preprocess.py then train_model.py to"
      " unlock this."
  )

  # ---- MACRO: network-wide stockout risk overview, all kiosks ----
  with tab_risk:
    if forecast_engine is None:
      st.warning(model_unavailable_msg)
    else:
      st.caption("Days of stock left = current stock ÷ predicted next-day demand.")
      with st.spinner("Scoring stockout risk across all kiosks..."):
        network_rows = []
        for hub in locations_df["Location_Name"].tolist():
          risk_df = forecast_engine.compute_stock_risk_panel(
              inventory[hub], hub, low_stock_threshold=LOW_STOCK_THRESHOLD_UNIVERSAL
          )
          network_rows.append({
              "Station": hub,
              "🔴 High Risk": (risk_df["ML_Demand_Risk_Status"] == "🔴 HIGH RISK (Stockout Imminent)").sum(),
              "🟡 Moderate": (risk_df["ML_Demand_Risk_Status"] == "🟡 MODERATE RISK (Reorder Soon)").sum(),
              "🟢 Healthy": (risk_df["ML_Demand_Risk_Status"] == "🟢 HEALTHY").sum(),
          })
        network_df = pd.DataFrame(network_rows).sort_values(
            "🔴 High Risk", ascending=False
        )
      st.dataframe(network_df, use_container_width=True, hide_index=True)

  # ---- MICRO: drill down into one kiosk's risk panel + 7-day forecast ----
  with tab_detail:
    if forecast_engine is None:
      st.warning(model_unavailable_msg)
    else:
      drilldown_hub = st.selectbox(
          "Station", locations_df["Location_Name"].tolist(), key="ml_drilldown_hub"
      )
      drilldown_risk_df = forecast_engine.compute_stock_risk_panel(
          inventory[drilldown_hub], drilldown_hub,
          low_stock_threshold=LOW_STOCK_THRESHOLD_UNIVERSAL,
      )
      st.dataframe(drilldown_risk_df, use_container_width=True, hide_index=True)

      st.markdown("---")
      forecast_sku = st.selectbox(
          "SKU to forecast",
          catalog_df["SKU"].tolist(),
          format_func=lambda s: get_field(
              catalog_df[catalog_df["SKU"] == s].iloc[0],
              ["Active_Ingredient_Description"],
              default=s,
          ),
          key="ml_forecast_sku",
      )
      forecasts = forecast_engine.predict_recursive_7_day(drilldown_hub, forecast_sku)
      if not forecasts:
        st.info("Not enough history for this SKU/station yet.")
      else:
        forecast_df = pd.DataFrame(forecasts)
        st.line_chart(forecast_df.set_index("Date")["Predicted_Demand"], height=200)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.caption("Days 1-2: real data. Days 3-7: directional outlook, not a firm number.")

  # ---- Model evaluation, filterable by segment (held-out only) ----
  with tab_eval:
    if forecast_engine is None:
      st.warning(model_unavailable_msg)
    else:
      st.caption("Held-out test data only — not evaluated on rows the model trained on.")
      station_eval_df, sku_eval_df = forecast_engine.get_per_group_evaluation()
      eval_segment_choice = st.radio(
          "Break down by:", ["Station", "SKU"], horizontal=True, key="eval_segment_choice"
      )
      eval_view = (
          station_eval_df.sort_values("RMSE") if eval_segment_choice == "Station"
          else sku_eval_df.sort_values("RMSE")
      )
      st.dataframe(eval_view, use_container_width=True, hide_index=True)

  # ---- Seasonal correlation and anomaly detection don't need the trained
  # model at all, so they stay available even before it's been trained. ----
  with tab_seasonal:
    seasonal = seasonal_category_insights(sales_df, catalog_df)
    top_spikes = seasonal.sort_values("Spike_Ratio", ascending=False).head(8)
    st.dataframe(
        top_spikes.rename(columns={
            "Symptom_Category": "Category",
            "Units_Sold": "Avg Daily Units (This Month)",
            "Overall_Avg": "Avg Daily Units (Yearly)",
            "Spike_Ratio": "Spike vs Yearly Avg",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Above 1.0x = sells more than usual that month — pre-stock ahead of it.")

  with tab_anomaly:
    purchases = pd.DataFrame(
        [r for r in audit_log if r.get("event_type") == "purchase"]
    )
    if purchases.empty:
      st.info("No purchases logged yet this session.")
    else:
      anomalies = detect_anomalies(sales_df, purchases)
      if anomalies.empty:
        st.success("No anomalies detected.")
      else:
        st.dataframe(
            anomalies[
                ["timestamp", "station_id", "sku", "quantity", "mean", "std", "z_score"]
            ].rename(columns={
                "mean": "Historical Avg/Day",
                "std": "Historical StdDev",
                "z_score": "Z-Score",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Flagged: quantity is 2+ standard deviations from that SKU/station's usual daily average.")


# Custom Deep Forest Green Background & Scoped CSS
st.markdown(
    """
    <style>
    /* Main App Deep Forest Green Background */
    .stApp {
        background-color: #0A3A1A !important;
    }

    /* General App Headings and Instructions in White.
       h4-h6 included because st.markdown("#### ...") renders <h4>, which
       was previously uncovered -- those headers fell back to Streamlit's
       default theme color and were nearly invisible against this dark
       background. ul/ol/li included because markdown bullet lists (e.g.
       the "N item(s) below threshold" alerts) render as <li> elements,
       which this rule didn't cover -- they fell back to Streamlit's
       default low-contrast gray, nearly invisible on this dark green
       background. */
    h1, h2, h3, h4, h5, h6, p, label, ul, ol, li {
        color: #FFFFFF !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Sidebar was previously unstyled and fell back to Streamlit's default
       light theme, while the rule above still forced near-white text onto
       it -- white-on-light-grey, unreadable. */
    section[data-testid="stSidebar"] {
        background-color: #072D14 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    /* The blanket white-text rule above also lands on the PIN/text input
       boxes, which keep their default white background -- white text on a
       white box is invisible while typing. Force dark, readable text (and
       an explicit white background) specifically inside input elements. */
    section[data-testid="stSidebar"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #FFFFFF !important;
        caret-color: #000000 !important;
    }

    /* Large Touch Buttons / Tiles */
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #145A28 !important;
        border-radius: 12px;
        padding: 18px 24px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        width: 100%;
        /* Streamlit's default button keeps text on one line and clips the
           overflow, which cut off longer names like "Sikkim (Gangtok
           Station)" and "Udaipur City Station". Let it wrap instead. */
        white-space: normal !important;
        height: auto !important;
        min-height: 3.2em;
    }

    .stButton>button p, .stButton>button div, .stButton>button span {
        color: #000000 !important;
        white-space: normal !important;
    }

    .stButton>button:hover {
        background-color: #E8F5E9 !important;
        border-color: #0A3A1A !important;
        color: #000000 !important;
    }

    .stButton>button[kind="primary"] {
        background-color: #145A28 !important;
        color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
    }

    .stButton>button[kind="primary"] p, .stButton>button[kind="primary"] span {
        color: #FFFFFF !important;
    }

    /* Tomato Red Kiosk Alert Banner */
    .kiosk-alert-banner {
        background-color: #D32F2F;
        color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }

    /* Product / order cards: the "h1,h2,h3,p,label {color:#FFFFFF !important}"
       rule above beats plain inline color styles (an !important declaration
       always wins over a non-important one, regardless of specificity).
       This more specific selector (class + element) is required to force
       black text back on for anything inside a white card. */
    .kiosk-product-card,
    .kiosk-product-card h3,
    .kiosk-product-card p,
    .kiosk-product-card b,
    .kiosk-product-card span {
        color: #000000 !important;
    }

    /* Status pill badges: standardized Critical/Warning/Stable chips so
       severity is scannable by shape+color, not just emoji or text alone. */
    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        color: #FFFFFF !important;
        white-space: nowrap;
    }
    .status-pill-critical { background-color: #D32F2F; }
    .status-pill-warning  { background-color: #B8860B; }
    .status-pill-stable   { background-color: #1B5E20; }

    /* Alert item cards: light, color-coded fills instead of plain text on
       the dark background, so severity is visible at a glance. */
    .alert-item-card {
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .alert-item-card.critical {
        background-color: #FBE2E1;
        border-left: 5px solid #D32F2F;
    }
    .alert-item-card.warning {
        background-color: #FCEFD1;
        border-left: 5px solid #B8860B;
    }
    .alert-item-card * {
        color: #1a1a1a !important;
    }
    .alert-item-card .alert-item-label {
        font-weight: 700;
    }
    .alert-item-card .alert-item-meta {
        font-size: 13px;
        color: #4a4a4a !important;
    }

    /* Tabs (both the role-level tabs -- Inventory & Alerts / Audit Log /
       ML Insights -- and the ML Insights sub-tabs) previously rendered as
       plain low-contrast text with just a thin underline on the active
       one, so they read as headings rather than something clickable.
       Give each tab a pill-button look, with a clearly distinct filled
       state for whichever one is active. */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: 2px solid rgba(255,255,255,0.15) !important;
    }
    .stTabs [role="tab"] {
        background-color: rgba(255,255,255,0.10) !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        color: #E8F5E9 !important;
    }
    .stTabs [role="tab"]:hover {
        background-color: rgba(255,255,255,0.20) !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #145A28 !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid #D32F2F !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def get_field(row, candidates, default="N/A"):
  """Return the first non-null, non-empty value among candidate columns.

  Using an explicit pd.notna()/strip() check (instead of `or`-chaining)
  matters here because pandas NaN is truthy in Python, so `row.get(col) or
  fallback` silently returns NaN instead of falling through to the next
  candidate column.
  """
  for col in candidates:
    if col in row.index:
      val = row[col]
      if pd.notna(val) and str(val).strip() != "":
        return val
  return default


@st.cache_data
def load_data():
  catalog_df = pd.read_csv("kiosk_product_catalog.csv")
  locations_df = pd.read_csv("kiosk_locations_config.csv")
  try:
    sales_df = pd.read_csv("historical_sales.csv")
  except FileNotFoundError:
    import generate_data

    generate_data.generate_synthetic_sales()
    sales_df = pd.read_csv("historical_sales.csv")
  return catalog_df, locations_df, sales_df


@st.cache_resource
def get_shared_inventory(_catalog_df, _locations_df):
  """Mock stock levels, shared across every session on this server process.

  This MUST be st.cache_resource (not st.session_state / st.cache_data).
  st.session_state is scoped to one browser tab, so a regional manager
  restocking on their own kiosk terminal would never be visible to the
  super admin or to a traveler on a different kiosk. cache_resource
  returns the same mutable dict object to every session, so a restock in
  one session is immediately visible to all others on their next rerun.
  Leading underscores on the params tell Streamlit not to hash these
  DataFrames for the cache key (they're only used once, at first build).
  """
  inventory = {}
  for _, loc_row in _locations_df.iterrows():
    loc_name = loc_row["Location_Name"]
    inventory[loc_name] = {}
    for _, cat_row in _catalog_df.iterrows():
      sku = cat_row["SKU"]
      # Seed a couple of SKUs below threshold so the alert flow is visible
      # on first run of the demo.
      inventory[loc_name][sku] = 8 if sku in ("AID-001", "DIG-002") else 45
  return inventory


catalog_df, locations_df, sales_df = load_data()
burn_rate_table = get_burn_rate_table(sales_df)
inventory = get_shared_inventory(catalog_df, locations_df)
audit_log = get_audit_log()


@st.cache_resource
def load_forecast_engine():
  """Cached as a resource since it wraps a live model object, not a
  serializable table. Returns None if the model hasn't been trained yet
  (train_model.py not run), so the app degrades gracefully instead of
  crashing on import."""
  try:
    return KioskForecastEngine()
  except FileNotFoundError:
    return None


forecast_engine = load_forecast_engine()

# Prototype-only PINs. These are plaintext constants purely to demonstrate
# role separation for the portfolio build; a real deployment would check
# hashed credentials (e.g. via st.secrets or a proper auth provider), never
# a hardcoded string compared in the app's own source.
OPERATOR_PIN = "1234"
MANAGER_PIN = "9999"


# Initialize session state
if "kiosk_step" not in st.session_state:
  st.session_state.kiosk_step = "welcome"
if "selected_location" not in st.session_state:
  st.session_state.selected_location = None
if "selected_symptom" not in st.session_state:
  st.session_state.selected_symptom = None
if "operator_logged_in" not in st.session_state:
  st.session_state.operator_logged_in = False
if "operator_station" not in st.session_state:
  st.session_state.operator_station = None
if "manager_logged_in" not in st.session_state:
  st.session_state.manager_logged_in = False
if "checkout_stage" not in st.session_state:
  st.session_state.checkout_stage = "view_items"
if "cart_item" not in st.session_state:
  st.session_state.cart_item = None

# ==================== SIDEBAR: ROLE NAVIGATION ====================
st.sidebar.title("🧭 Navigation")
view_mode = st.sidebar.radio(
    "View as:",
    ["🧳 Traveler", "🔧 Operator", "🏙️ Manager"],
    key="view_mode",
)

if view_mode == "🔧 Operator":
  st.sidebar.markdown("---")
  if st.session_state.operator_logged_in:
    st.sidebar.success(f"Logged in — {st.session_state.operator_station}")
    if st.sidebar.button("🚪 Logout Operator"):
      st.session_state.operator_logged_in = False
      st.session_state.operator_station = None
      st.rerun()
  else:
    st.sidebar.markdown("**Operator Login**")
    op_station = st.sidebar.selectbox(
        "Your Station", locations_df["Location_Name"].tolist(), key="op_station_select"
    )
    op_pin = st.sidebar.text_input(
        "PIN", type="password", key="operator_pin_input"
    )
    if st.sidebar.button("Login", key="operator_login_btn"):
      if op_pin == OPERATOR_PIN:
        st.session_state.operator_logged_in = True
        st.session_state.operator_station = op_station
        st.rerun()
      else:
        st.sidebar.error("Incorrect PIN.")

elif view_mode == "🏙️ Manager":
  st.sidebar.markdown("---")
  if st.session_state.manager_logged_in:
    st.sidebar.success("Logged in — Network Manager")
    if st.sidebar.button("🚪 Logout Manager"):
      st.session_state.manager_logged_in = False
      st.rerun()
  else:
    st.sidebar.markdown("**Manager Login**")
    mgr_pin = st.sidebar.text_input(
        "PIN", type="password", key="manager_pin_input"
    )
    if st.sidebar.button("Login", key="manager_login_btn"):
      if mgr_pin == MANAGER_PIN:
        st.session_state.manager_logged_in = True
        st.rerun()
      else:
        st.sidebar.error("Incorrect PIN.")

# Header banner. Wrapped in its own bordered/shadowed card (a different
# shade than the page background) so it reads as a persistent top bar
# across every view, instead of blending into the same green as the body
# content below it.
st.markdown(
    """
    <div style="background-color: #072D14; padding: 22px 28px; border-radius: 14px; margin-bottom: 24px; border: 1px solid #145A28; box-shadow: 0 6px 14px rgba(0,0,0,0.35);">
      <div style="display: flex; align-items: center; gap: 15px;">
          <span style="background-color: #D32F2F; color: #FFFFFF; font-size: 36px; font-weight: bold; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">+</span>
          <h1 style="color: #FFFFFF; font-size: 36px; margin: 0; line-height: 1.1;">Transit Emergency Medicine Kiosk</h1>
      </div>
      <p style="color: #E8F5E9; font-size: 16px; margin: 8px 0 0 75px;">Automated touch dispensary for rapid, location-aware travel health relief.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== OPERATOR VIEW ====================
# Single assigned station only: sees the low-stock alert for their own
# kiosk and can restock it. No cross-station visibility, no ML tab -- kept
# deliberately simple, matching a single physical terminal's scope.
if view_mode == "🔧 Operator":
  if not st.session_state.operator_logged_in:
    st.info("Enter your station and PIN in the sidebar to access the Operator Portal.")
  else:
    st.header("Operator Portal")
    render_alerts_and_restock_tab(
        hubs_to_check=[st.session_state.operator_station],
        restock_hub_choices=[st.session_state.operator_station],
        actor_label=f"Operator: {st.session_state.operator_station}",
    )

# ==================== MANAGER VIEW ====================
# Full network scope by default (every kiosk/city at once) plus the audit
# trail and ML forecasting/risk tabs -- the macro-to-micro decision-support
# view, as opposed to the Operator's single-station action view.
elif view_mode == "🏙️ Manager":
  if not st.session_state.manager_logged_in:
    st.info("Enter the Manager PIN in the sidebar to access the Manager Portal.")
  else:
    st.header("Network Manager Portal — All Kiosks")
    st.caption("Every kiosk, ML demand forecasts, and stockout risk in one place.")

    tab_alerts, tab_audit, tab_ml = st.tabs(
        ["Inventory & Alerts", "Audit Log", "ML Insights"]
    )

    with tab_alerts:
      render_alerts_and_restock_tab(
          hubs_to_check=locations_df["Location_Name"].tolist(),
          restock_hub_choices=locations_df["Location_Name"].tolist(),
          actor_label="Manager (Network)",
      )

    with tab_audit:
      render_audit_log_tab()

    with tab_ml:
      render_ml_insights_tab()


# ==================== TRAVELER TOUCH KIOSK VIEW ====================
else:
  if st.session_state.kiosk_step == "welcome":
    st.markdown(
        "<div style='text-align: center; padding: 40px;'>"
        "<h1 style='font-size: 42px; margin-bottom: 20px; color: #FFFFFF;'>Feeling"
        " Unwell While Traveling?</h1>"
        "<p style='font-size: 20px; color: #E8F5E9; margin-bottom: 40px;'>Get"
        " instant, climate-optimized emergency relief tailored specifically"
        " to your transit hub.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    col_w1, col_w2, col_w3 = st.columns([1, 2, 1])
    with col_w2:
      if st.button("Touch Here to Begin", type="primary", use_container_width=True):
        st.session_state.kiosk_step = "select_location"
        st.rerun()

  elif st.session_state.kiosk_step == "select_location":
    st.subheader("Step 1 of 2: Select Your Transit Hub")
    locations_list = locations_df["Location_Name"].tolist()
    loc_cols = st.columns(3)

    for idx, loc_name in enumerate(locations_list):
      col_idx = idx % 3
      loc_row = locations_df[locations_df["Location_Name"] == loc_name].iloc[0]
      with loc_cols[col_idx]:
        if st.button(
            f"📍 {loc_name}\n\n({loc_row['Sub_Group']})", key=f"loc_tile_{idx}"
        ):
          st.session_state.selected_location = loc_name
          st.session_state.kiosk_step = "select_symptom"
          st.rerun()

    st.markdown("---")
    if st.button("← Back to Welcome"):
      st.session_state.kiosk_step = "welcome"
      st.rerun()

  elif st.session_state.kiosk_step == "select_symptom":
    loc_row = locations_df[
        locations_df["Location_Name"] == st.session_state.selected_location
    ].iloc[0]
    st.markdown(
        f"""
        <div class="kiosk-alert-banner">
            📍 Hub: {st.session_state.selected_location} ({loc_row['Sub_Group']})<br>
            <span style="font-size: 18px; font-weight: normal;">Primary Risk Factor: {loc_row['Primary_Risk']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Step 2 of 2: What discomfort are you experiencing?")
    symptoms_list = catalog_df["Symptom_Category"].unique().tolist()
    sym_cols = st.columns(3)

    for idx, sym in enumerate(symptoms_list):
      col_idx = idx % 3
      with sym_cols[col_idx]:
        if st.button(f"💊\n\n{sym}", key=f"sym_tile_{idx}kec"):
          log_event(
              "symptom_selected",
              station_id=st.session_state.selected_location,
              sub_group=loc_row["Sub_Group"],
              primary_risk=loc_row["Primary_Risk"],
              symptom_category=sym,
              actor="customer",
          )
          st.session_state.selected_symptom = sym
          # Reset the checkout flow so a stale "success"/"payment" stage
          # from a previous item never leaks into a fresh symptom pick.
          st.session_state.checkout_stage = "view_items"
          st.session_state.cart_item = None
          st.session_state.kiosk_step = "dispense"
          st.rerun()

    st.markdown("---")
    if st.button("← Change Location"):
      st.session_state.kiosk_step = "select_location"
      st.rerun()

  elif st.session_state.kiosk_step == "dispense":
    loc_row = locations_df[
        locations_df["Location_Name"] == st.session_state.selected_location
    ].iloc[0]
    sub_group = loc_row["Sub_Group"]
    selected_location = st.session_state.selected_location
    selected_symptom = st.session_state.selected_symptom
    region_critical_skus = get_region_critical_skus(
        selected_location, catalog_df, locations_df
    )

    st.markdown(
        f"""
        <div class="kiosk-alert-banner">
            Dispensing Recommendations for {selected_location} — {selected_symptom}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- STAGE 1: VIEW ITEMS & ADD TO CART ----------------
    if st.session_state.checkout_stage == "view_items":
      symptom_matches = catalog_df["Symptom_Category"] == selected_symptom
      region_matches = catalog_df["Target_Destination_Tag"].apply(
          lambda tag: tag.strip().lower() == "all"
          or is_region_specific_match(tag, sub_group, selected_location)
      )
      filtered_catalog = catalog_df[symptom_matches & region_matches]
      if filtered_catalog.empty:
        filtered_catalog = catalog_df[
            catalog_df["Symptom_Category"] == selected_symptom
        ]

      for _, row in filtered_catalog.iterrows():
        sku = get_field(row, ["SKU"], "N/A")
        desc = get_field(
            row,
            [
                "Active_Ingredient_Description",  # real column name
                "Product_Name",
                "Item_Description",
                "Description",
            ],
            default=sku,
        )
        form = get_field(row, ["Form_Factor", "Form", "Type"], default="Standard")
        unit_price = float(
            get_field(
                row,
                ["Unit_Price_INR", "Price", "Unit_Price"],  # real column name first
                default="0",
            )
        )
        stock_qty = inventory[selected_location].get(sku, 0)

        # Region-critical items (the reason this hub's kiosk exists, e.g.
        # motion sickness meds at a Hill Station) get a higher per-passenger
        # cap than generic "All" items, which are easier to hoard and more
        # replaceable elsewhere -- reuses the same region_critical_skus set
        # that drives the manager alert prioritization, so the two stay in
        # sync instead of encoding the rule twice.
        is_critical = sku in region_critical_skus
        max_per_passenger = 5 if is_critical else 3
        critical_badge = (
            '<span style="background-color: #D32F2F; color: #FFFFFF;'
            " padding: 3px 10px; border-radius: 6px; font-size: 13px;"
            ' font-weight: bold; margin-left: 10px;">🔥 REGION-CRITICAL</span>'
            if is_critical
            else ""
        )

        st.markdown(
            f"""
            <div class="kiosk-product-card" style="padding: 28px; border-radius: 16px; background-color: #ffffff; margin-bottom: 20px; border: 2px solid #145A28; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
                <h3 style="margin: 0 0 12px 0; font-size: 24px; font-weight: bold;">💊 {desc}{critical_badge}</h3>
                <p style="margin: 8px 0; font-size: 18px;"><b>Form Factor:</b> {form} &nbsp;|&nbsp; <b>Price:</b> ₹{unit_price:.0f}</p>
                <div style="margin-top: 12px;"><span style="background-color: #E8F5E9; padding: 6px 14px; border-radius: 8px; font-weight: bold; font-size: 15px; border: 1px solid #C8E6C9;">SKU Code: {sku} (Stock Available: {stock_qty})</span></div>
                <p style="margin: 10px 0 0 0; font-size: 14px; color: #555555 !important;">Max {max_per_passenger} units per passenger</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if stock_qty > 0:
          col_qty, col_btn = st.columns([1, 2])
          with col_qty:
            selected_qty = st.number_input(
                "Qty",
                min_value=1,
                max_value=min(max_per_passenger, stock_qty),
                value=1,
                key=f"qty_sel_{sku}",
            )
          with col_btn:
            st.write("")  # alignment spacing
            if st.button(f"🛒 Add to Cart: {sku}", key=f"add_cart_{sku}"):
              # Belt-and-suspenders: the number_input widget already caps
              # this, but a passenger safety limit shouldn't rely on a
              # single UI control never being bypassed.
              if selected_qty > max_per_passenger:
                st.error(
                    "You have exceeded the maximum purchase limit for this"
                    f" item ({max_per_passenger} units max allowed per"
                    " passenger)."
                )
              elif selected_qty > stock_qty:
                st.error("Not enough stock available at this station.")
              else:
                st.session_state.cart_item = {
                    "desc": desc,
                    "sku": sku,
                    "unit_price": unit_price,
                    "quantity": selected_qty,
                    "price": unit_price * selected_qty,
                    "region_critical": is_critical,
                }
                st.session_state.checkout_stage = "order_summary"
                st.rerun()
        else:
          st.error("Out of Stock at this station.")

      st.markdown("---")
      col_d1, col_d2 = st.columns(2)
      with col_d1:
        if st.button("Start Over"):
          st.session_state.kiosk_step = "welcome"
          st.session_state.checkout_stage = "view_items"
          st.session_state.cart_item = None
          st.rerun()
      with col_d2:
        if st.button("Change Symptom"):
          st.session_state.kiosk_step = "select_symptom"
          st.session_state.checkout_stage = "view_items"
          st.session_state.cart_item = None
          st.rerun()

    # ---------------- STAGE 2: ORDER NOW (CART REVIEW) ----------------
    elif st.session_state.checkout_stage == "order_summary":
      item = st.session_state.cart_item
      st.subheader("📦 Order Summary")
      badge = " 🔥 Region-Critical" if item.get("region_critical") else ""
      st.markdown(
          f"""
            <div class="kiosk-product-card" style="padding: 24px; border-radius: 12px; background-color: #ffffff; border: 2px solid #145A28; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">{item['desc']}{badge}</h3>
                <p style="font-size: 18px;"><b>SKU:</b> {item['sku']} &nbsp;|&nbsp; <b>Quantity:</b> {item['quantity']} &nbsp;|&nbsp; <b>Unit Price:</b> ₹{item['unit_price']:.0f}</p>
                <p style="font-size: 20px; font-weight: bold;">Total Amount: ₹{item['price']:.0f}</p>
            </div>
            """,
          unsafe_allow_html=True,
      )

      col_o1, col_o2 = st.columns(2)
      with col_o1:
        if st.button("Proceed to Payment", type="primary"):
          st.session_state.checkout_stage = "payment"
          st.rerun()
      with col_o2:
        if st.button("← Back to Catalog"):
          st.session_state.checkout_stage = "view_items"
          st.rerun()

    # ---------------- STAGE 3: PROCEED TO PAYMENT ----------------
    elif st.session_state.checkout_stage == "payment":
      item = st.session_state.cart_item
      st.subheader("💳 Mock Payment Gateway")
      st.markdown(
          f"<p style='font-size: 18px;'>Amount to Pay: <b>₹{item['price']:.0f}</b></p>",
          unsafe_allow_html=True,
      )
      st.info(
          "Mock Terminal: Scan QR Code or Tap Card below to simulate secure"
          " payment."
      )

      # Stock is now shared live across every manager/kiosk session, so
      # re-check right before confirming in case someone else bought the
      # last unit (or a manager zeroed it out) while this customer was
      # mid-checkout.
      live_stock = inventory[selected_location].get(item["sku"], 0)
      if live_stock < item["quantity"]:
        st.error(
            "Stock at this station dropped below your requested quantity"
            " while you were checking out. Please go back and adjust your"
            " order."
        )
        if st.button("← Back to Catalog"):
          st.session_state.checkout_stage = "view_items"
          st.rerun()
      else:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
          if st.button(
              "Simulate Successful Payment", type="primary", key="pay_sim_btn"
          ):
            sku_purchased = item["sku"]
            current_stock = inventory[selected_location].get(sku_purchased, 0)
            new_stock = max(0, current_stock - item["quantity"])
            inventory[selected_location][sku_purchased] = new_stock
            log_event(
                "purchase",
                station_id=selected_location,
                sub_group=sub_group,
                primary_risk=loc_row["Primary_Risk"],
                symptom_category=selected_symptom,
                sku=sku_purchased,
                quantity=item["quantity"],
                unit_price=item["unit_price"],
                total_price=item["price"],
                region_critical=item.get("region_critical", False),
                stock_before=current_stock,
                stock_after=new_stock,
                actor="customer",
            )
            st.session_state.checkout_stage = "success"
            st.rerun()
        with col_p2:
          if st.button("Cancel / Back"):
            st.session_state.checkout_stage = "order_summary"
            st.rerun()

    # ---------------- STAGE 4: SUCCESSFULLY PAID & COLLECT ORDER --------
    elif st.session_state.checkout_stage == "success":
      st.markdown(
          """
            <div style="background-color: #1B5E20; color: #FFFFFF; padding: 30px; border-radius: 16px; text-align: center; margin-bottom: 25px;">
                <h2 style="color: #FFFFFF; margin-top: 0;">🎉 Successfully Paid!</h2>
                <p style="font-size: 20px; margin-bottom: 0;">Your payment was processed successfully. Please collect your item from the lower kiosk tray.</p>
            </div>
            """,
          unsafe_allow_html=True,
      )

      if st.button("📥 Collect Order & Finish", type="primary"):
        st.session_state.kiosk_step = "welcome"
        st.session_state.checkout_stage = "view_items"
        st.session_state.cart_item = None
        st.rerun()

# py -m streamlit run streamlit_app.py
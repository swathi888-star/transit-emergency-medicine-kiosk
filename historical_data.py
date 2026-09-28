from datetime import datetime, timedelta
import numpy as np
import pandas as pd


CONFIG_PATH = "kiosk_locations_config.csv"
CATALOG_PATH = "kiosk_product_catalog.csv"


def load_stations(filepath=CONFIG_PATH):
  """Loads all 13 live kiosk locations from kiosk_locations_config.csv."""
  df = pd.read_csv(filepath)
  return df.to_dict("records")


def load_skus(filepath=CATALOG_PATH):
  """Loads all 11 live SKUs from kiosk_product_catalog.csv."""
  df = pd.read_csv(filepath)
  return df.to_dict("records")


def sku_is_relevant(tag, station):
  """Whether a SKU's Target_Destination_Tag plausibly applies at this station.

  Tags in the real catalog are free text ("All", "Delhi", "Hill Stations",
  "Coastal Hubs & Hill Stations", "Arid Deserts & Hyderabad", ...), so this
  matches on keywords against the station's Sub_Group / Category_Type / Name
  rather than requiring an exact string match.
  """
  tag_l = str(tag).lower()
  if tag_l == "all":
    return True

  name_l = station["Location_Name"].lower()
  sub_group_l = station["Sub_Group"].lower()
  category_l = station["Category_Type"].lower()

  keyword_checks = [
      ("hill" in tag_l and "hill" in sub_group_l),
      ("coastal" in tag_l and "coastal" in sub_group_l),
      ("humid" in tag_l and sub_group_l in ("coastal & beach", "coastal metros")),
      ("metro" in tag_l and "urban metro" in category_l),
      ("arid" in tag_l and "arid" in sub_group_l),
      ("desert" in tag_l and "arid" in sub_group_l),
      ("delhi" in tag_l and "delhi" in name_l),
      ("bengaluru" in tag_l and "bengaluru" in name_l),
      ("hyderabad" in tag_l and "hyderabad" in name_l),
  ]
  return any(keyword_checks)


def is_heat_or_hydration_sku(symptom_category):
  cat_l = str(symptom_category).lower()
  return "dehydration" in cat_l or "heat exhaustion" in cat_l


def is_respiratory_or_digestive_sku(symptom_category):
  cat_l = str(symptom_category).lower()
  return "respiratory" in cat_l or "dehydration" in cat_l or "diarrhea" in cat_l.replace(
      "diarrhoea", "diarrhea"
  ) or "acidity" in cat_l


def generate_raw_logs():
  # Set a consistent random seed for reproducibility of chaos
  np.random.seed(42)

  # Stations and SKUs are now pulled directly from the live kiosk system's
  # kiosk_locations_config.csv (13 stations) and kiosk_product_catalog.csv
  # (11 SKUs), so the simulated telemetry covers every real kiosk/SKU pair
  # instead of a hardcoded 4-station / 6-SKU subset.
  stations = load_stations()
  sku_records = load_skus()
  skus = [s["SKU"] for s in sku_records]
  sku_lookup = {s["SKU"]: s for s in sku_records}

  # 1. Comprehensive Festival & Major Holiday Windows (2024 - 2025)
  festival_ranges = [
      pd.date_range(start="2024-04-10", end="2024-04-14"),  # Spring / Regional festivals
      pd.date_range(start="2024-10-09", end="2024-10-13"),  # Dussehra 2024
      pd.date_range(start="2024-10-29", end="2024-11-03"),  # Diwali 2024
      pd.date_range(start="2025-03-30", end="2025-04-03"),  # Spring festivals 2025
      pd.date_range(start="2025-09-28", end="2025-10-02"),  # Dussehra 2025
      pd.date_range(start="2025-10-18", end="2025-10-23"),  # Diwali 2025
  ]
  all_festival_dates = set()
  for dr in festival_ranges:
    all_festival_dates.update(dr.date)

  all_dates = pd.date_range(start="2024-01-01", end="2025-12-31")

  # 2a. Local "Black Swan" Transit Surge Days per Station (unannounced local
  # delays, rallies, station-specific disruptions) - independent per station.
  station_black_swan_days = {}
  for station in stations:
    swans = np.random.choice(all_dates, size=12, replace=False)
    station_black_swan_days[station["Location_Name"]] = set(
        pd.Timestamp(d).date() for d in swans
    )

  # 2b. National-level uncertainty events (e.g. a nationwide flight/rail
  # delay wave) that hit every station, and every SKU at that station, on
  # the same day - distinct from the per-station shocks above.
  national_swans = np.random.choice(all_dates, size=6, replace=False)
  national_black_swan_days = set(pd.Timestamp(d).date() for d in national_swans)

  # 3. Localized Viral/Health Outbreak Windows (3-to-4 day clusters),
  # generated independently per station so health issues genuinely vary
  # by city instead of every station catching the same outbreak on the
  # same dates.
  station_outbreak_days = {}
  for station in stations:
    outbreak_dates = set()
    for _ in range(2):  # two outbreak windows per station over the 2 years
      window_start = pd.Timestamp(np.random.choice(all_dates))
      window_len = int(np.random.choice([3, 4]))
      outbreak_dates.update(
          pd.date_range(start=window_start, periods=window_len).date
      )
    station_outbreak_days[station["Location_Name"]] = outbreak_dates

  start_date = datetime(2024, 1, 1)
  end_date = datetime(2025, 12, 31)
  current_date = start_date

  raw_events = []

  while current_date <= end_date:
    dt_date = current_date.date()
    month = current_date.month
    day = current_date.day
    # Fixed: Python datetime uses .weekday() where >= 4 means Friday, Saturday, Sunday
    is_weekend = current_date.weekday() >= 4
    is_festival = dt_date in all_festival_dates

    # Seasonal Multiplier Flags
    is_summer_break = month in [4, 5, 6]  # April to June travel season
    is_winter_break = (month == 12 and day >= 15) or (month == 1 and day <= 5)  # Year-end holiday travel

    for station in stations:
      station_name = station["Location_Name"]
      sub_group = station["Sub_Group"]
      station_base_multiplier = float(station["Base_Multiplier"])
      is_black_swan = (
          dt_date in station_black_swan_days[station_name]
          or dt_date in national_black_swan_days
      )
      is_outbreak = dt_date in station_outbreak_days[station_name]

      # Climatic uncertainty: 8% daily chance of an unseasonal weather blip
      unseasonal_weather_factor = (
          1.4 if np.random.rand() < 0.08 else 1.0
      )

      for sku in skus:
        sku_info = sku_lookup[sku]
        symptom_category = sku_info["Symptom_Category"]
        relevant = sku_is_relevant(sku_info["Target_Destination_Tag"], station)

        # Base transaction probability: a SKU that's actually targeted at
        # this kind of station sells at normal volume; an off-tag SKU
        # (e.g. a Delhi-tagged respiratory spray at a beach kiosk) still
        # sells occasionally to passing travelers, just far less often.
        base_lambda = 3.0 if relevant else 0.3

        # Station-level risk multiplier from the live config (e.g. Goa's
        # 2.0 vs Bengaluru's 1.4) scales demand at that specific kiosk.
        base_lambda *= station_base_multiplier

        # Apply deterministic calendar rules & seasonal bumps
        if is_weekend:
          base_lambda *= 1.3
        if is_festival:
          base_lambda *= 1.8
        if is_summer_break:
          base_lambda *= 1.25
        if is_winter_break:
          base_lambda *= 1.35
        if (
            sub_group in ("Coastal & Beach", "Coastal Metros")
            and is_heat_or_hydration_sku(symptom_category)
            and month in [4, 5, 6]
        ):
          base_lambda *= 1.6

        # --- APPLY UNCERTAINTY & CHAOS MULTIPLIERS ---
        # A. Unannounced Black Swan transit shock
        if is_black_swan:
          base_lambda *= np.random.uniform(2.0, 2.8)

        # B. Climatic weather shock (e.g., sudden heatwave/cold snap)
        base_lambda *= unseasonal_weather_factor

        # C. Localized behavioral outbreak shock (affects respiratory/digestion)
        if is_outbreak and is_respiratory_or_digestive_sku(symptom_category):
          base_lambda *= 2.2

        # Simulate number of individual purchases for this SKU today using Poisson noise
        num_sales = np.random.poisson(base_lambda)

        for _ in range(num_sales):
          # Random operating hour (6 AM to 11 PM)
          random_hour = np.random.randint(6, 23)
          random_minute = np.random.randint(0, 60)
          random_second = np.random.randint(0, 60)
          event_time = current_date + timedelta(
              hours=random_hour, minutes=random_minute, seconds=random_second
          )

          # REAL-WORLD IOT GLITCH 1: Network drop / delayed sync (2% delayed by 1-2 days)
          if np.random.rand() < 0.02:
            event_time += timedelta(days=int(np.random.choice([1, 2])))

          raw_events.append({
              "Timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
              "Station_ID": station_name,
              "Sub_Group": sub_group,
              "SKU": sku,
              "Quantity_Dispensed": np.random.choice(
                  [1, 2], p=[0.85, 0.15]
              ),  # Safety guardrail adherence
              "Transaction_Status": "SUCCESS",
          })

    current_date += timedelta(days=1)

  df_raw = pd.DataFrame(raw_events)

  # REAL-WORLD IOT GLITCH 2: Missing Data & Duplicates
  df_raw = pd.concat(
      [df_raw, df_raw.sample(n=150, random_state=42)], ignore_index=True
  )
  if not df_raw.empty:
    null_indices = np.random.choice(df_raw.index, size=50, replace=False)
    df_raw.loc[null_indices, "Station_ID"] = np.nan

  df_raw.to_csv("raw_vending_telemetry.csv", index=False)
  print(
      f"Generated chaotic raw telemetry log with {len(df_raw):,} records"
      f" across {len(stations)} stations and {len(skus)} SKUs, containing"
      " full seasonal ranges, black-swan surges, and weather anomalies."
  )


if __name__ == "__main__":
  generate_raw_logs()

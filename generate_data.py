from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_sales(
    catalog_path='kiosk_product_catalog.csv',
    locations_path='kiosk_locations_config.csv',
    output_path='historical_sales.csv',
):
  catalog = pd.read_csv(catalog_path)
  locations_df = pd.read_csv(locations_path)

  start_date = datetime(2025, 10, 1)
  end_date = datetime(2026, 3, 31)
  current_date = start_date

  sales_records = []
  np.random.seed(42)

  while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')

    for _, loc in locations_df.iterrows():
      kiosk_name = loc['Location_Name']
      sub_group = loc['Sub_Group']
      category_type = loc['Category_Type']
      base_mult = loc['Base_Multiplier']

      footfall = np.random.randint(900, 2600)

      for _, row in catalog.iterrows():
        sku = row['SKU']
        category = row['Symptom_Category']
        target_tag = row['Target_Destination_Tag']

        # Base distribution
        units = (
            np.random.poisson(lam=3)
            if target_tag == 'All'
            else np.random.poisson(lam=1)
        )

        # Match logic for tiered destination tagging
        is_match = False
        if target_tag == 'All':
          is_match = True
        elif (
            target_tag in kiosk_name
            or target_tag == sub_group
            or target_tag in sub_group
        ):
          is_match = True
        elif category_type == 'Urban Metro' and target_tag in kiosk_name:
          is_match = True

        if is_match:
          units = int(units * base_mult * np.random.uniform(1.2, 1.8))

        # Winter boost for cough/respiratory in north/high-altitude hubs
        if current_date.month in [12, 1] and (
            'Cough' in category or 'Respiratory' in category
        ):
          if (
              'Delhi' in kiosk_name
              or 'Leh' in kiosk_name
              or 'Sikkim' in kiosk_name
              or 'Manali' in kiosk_name
          ):
            units = int(units * 2.2)

        sales_records.append({
            'Date': date_str,
            'Kiosk_Location': kiosk_name,
            'Sub_Group': sub_group,
            'SKU': sku,
            'Symptom_Category': category,
            'Units_Sold': max(0, units),
            'Footfall_Volume': footfall,
        })

    current_date += timedelta(days=1)

  sales_df = pd.DataFrame(sales_records)
  sales_df.to_csv(output_path, index=False)
  print(
      f'Generated {len(sales_df)} multi-tiered transaction records saved to'
      f' {output_path}'
  )


if __name__ == '__main__':
  generate_synthetic_sales()
import pandas as pd


def load_catalog(filepath='kiosk_product_catalog.csv'):
  """Loads the product catalog CSV into a Pandas DataFrame."""
  try:
    df = pd.read_csv(filepath)
    return df
  except FileNotFoundError:
    print(f'Error: {filepath} not found in the current directory.')
    return None


def get_recommendations(df, user_symptom, user_destination):
  """Filters the catalog based on symptom category and destination relevance."""
  # Filter by symptom match and ensure destination matches either 'All' or the specific location
  results = df[
      (df['Symptom_Category'].str.lower() == user_symptom.lower())
      & (
          (df['Target_Destination_Tag'].str.lower() == 'all')
          | (df['Target_Destination_Tag'].str.lower() == user_destination.lower())
      )
  ]

  # Fallback if no location-specific match is found but the symptom exists
  if results.empty:
    results = df[df['Symptom_Category'].str.lower() == user_symptom.lower()]

  return results


if __name__ == '__main__':
  catalog = load_catalog()

  if catalog is not None:
    print('--- Kiosk Catalog Loaded Successfully --- \n')

    # Test query: A traveler in a Hill Station looking for Pain relief
    test_symptom = 'Motion Sickness'
    test_destination = 'Leh'

    print(f'Query -> Symptom: {test_symptom} | Destination: {test_destination}\n')
    recommendations = get_recommendations(
        catalog, test_symptom, test_destination
    )

    if not recommendations.empty:
      print(recommendations[['SKU', 'Active_Ingredient_Description', 'Unit_Price_INR']])
    else:
      print('No matching medication found.')
"""Numeric type coercion for climate dataset columns."""
import pandas as pd


NUMERIC_COLUMNS = [
    'year',
    'global_avg_temperature', 'temperature_anomaly',
    'max_temperature', 'min_temperature',
    'sea_surface_temperature', 'sea_level_rise_mm', 'annual_rainfall_mm',
    'heatwave_days', 'drought_index', 'flood_events_count',
    'climate_risk_index',
    'co2_concentration_ppm', 'fossil_fuel_consumption',
    'renewable_energy_share', 'forest_cover_percent',
    'deforestation_rate', 'air_quality_index',
    'predicted_temperature_2050',
]


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns to numeric dtype, logging any conversion failures."""
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        after_na = df[col].isna().sum()
        if after_na > before_na:
            print(f"  {col}: {after_na - before_na} values failed numeric conversion")
    return df

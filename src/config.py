from pathlib import Path

# Base project directories 
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"   
FIGURES_DIR = BASE_DIR / "figures"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"
EDA_FIG_OPSD = FIGURES_DIR / "eda_opsd"
EDA_FIG_CO2 = FIGURES_DIR / "eda_co2"
EDA_FIG_WEATHER = FIGURES_DIR / "eda_weather"



for p in [PROCESSED_DIR, FIGURES_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# Raw dataset file paths 
OPSD_FILE = RAW_DIR / "time_series_60min_singleindex.csv"
CO2_FILE = RAW_DIR / "co2-emissions-per-capita-4.csv"
WEATHER_FILE = RAW_DIR / "wetter_und_camping.csv"

# OPSD dataset columns 
TIME_COL_OPSD = "utc_timestamp"
DE_LOAD_COL = "DE_load_actual_entsoe_transparency"
DE_SOLAR_COL = "DE_solar_generation_actual"
DE_WIND_ON_COL = "DE_wind_onshore_generation_actual"
DE_WIND_OFF_COL = "DE_wind_offshore_generation_actual"
DE_PRICE_COL = "DE_price_day_ahead"

# CO2 dataset columns 
CO2_YEAR_COL = "Year"
CO2_VALUE_COL = "Annual CO₂ emissions (per capita)"

# Weather dataset columns 
WEATHER_DATE_COL = "date"
TEMP_MEAN_COL = "mean_air_temp_mean"
PRECIP_COL = "mean_precipitation"
SUN_COL = "mean_sunshine_duration"

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


FIGURES_DIR = BASE_DIR / "figures"


# Create directories if missing
for p in [PROCESSED_DIR, FIGURES_DIR, EDA_FIG_OPSD, EDA_FIG_CO2, EDA_FIG_WEATHER]:
    p.mkdir(parents=True, exist_ok=True)
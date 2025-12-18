import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# We store all file paths and column names in 'config.py' to keep the code clean 
from src.config import (
    OPSD_FILE, CO2_FILE, WEATHER_FILE,
    TIME_COL_OPSD, WEATHER_DATE_COL
)

# This function loads the OPSD dataset and prepare the data for the times series analysis later on 
def load_opsd():
    df = pd.read_csv(OPSD_FILE, parse_dates=[TIME_COL_OPSD], low_memory=False)
    df = df.sort_values(TIME_COL_OPSD)
    return df

# This function loads Germany's Co2's emission dataset and ensure of the datatype of the 'Year' column so it can be used in regression analysis later on 
def load_co2_germany():
    df = pd.read_csv(CO2_FILE)
    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(int)
    return df

# This function loads the weather and camping dataset , convert the date to a datetime object and sort it by date for better analysis later 
def load_weather():
    df = pd.read_csv(WEATHER_FILE)
    if WEATHER_DATE_COL in df.columns:
        df[WEATHER_DATE_COL] = pd.to_datetime(df[WEATHER_DATE_COL])
        df = df.sort_values(WEATHER_DATE_COL)
    return df

# this is a sanity check , it prints the datasets shape to make sure they were loaded correctly 
if __name__ == "__main__":
    opsd = load_opsd()
    print("OPSD shape:", opsd.shape)
    co2 = load_co2_germany()
    print("CO2 shape:", co2.shape)
    weather = load_weather()
    print("Weather shape:", weather.shape)

import os
import sys
import numpy as np
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_PROCESSED_DIR,
    DE_LOAD_COL, DE_SOLAR_COL, DE_WIND_ON_COL, DE_WIND_OFF_COL,
    CO2_YEAR_COL, CO2_VALUE_COL
)

# 1) OPSD Feature Engineering


def load_opsd_preprocessed():
    df = pd.read_csv(DATA_PROCESSED_DIR / "opsd_daily_germany.csv", parse_dates=["utc_timestamp"])
    df = df.sort_values("utc_timestamp").set_index("utc_timestamp")
    return df

def add_opsd_features(df):
    df = df.copy()

    
    df["load_7d_avg"] = df[DE_LOAD_COL].rolling(7).mean()
    df["load_30d_avg"] = df[DE_LOAD_COL].rolling(30).mean()
    df["sol_7d_avg"] = df[DE_SOLAR_COL].rolling(7).mean()
    df["wind_on_7d_avg"] = df[DE_WIND_ON_COL].rolling(7).mean()
    df["wind_off_7d_avg"] = df[DE_WIND_OFF_COL].rolling(7).mean()

    
    df["load_lag1"] = df[DE_LOAD_COL].shift(1)
    df["load_lag7"] = df[DE_LOAD_COL].shift(7)

    
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna()
    return df

def build_opsd_features():
    df = load_opsd_preprocessed()
    df = add_opsd_features(df)

    out_path = DATA_PROCESSED_DIR / "opsd_features.csv"
    df.to_csv(out_path)
    print("Saved OPSD features to:", out_path)
    return df



# 2) CO₂ Feature Engineering

def load_co2_preprocessed():
    df = pd.read_csv(DATA_PROCESSED_DIR / "co2_germany_clean.csv")
    df = df.sort_values(CO2_YEAR_COL)
    df.rename(columns={CO2_VALUE_COL: "CO2"}, inplace=True)
    return df

def add_co2_features(df):
    df = df.copy()

    
    df["co2_lag1"] = df["CO2"].shift(1)
    df["co2_lag2"] = df["CO2"].shift(2)
    df["co2_lag5"] = df["CO2"].shift(5)

    
    df["co2_ma3"] = df["CO2"].rolling(3).mean()
    df["co2_ma5"] = df["CO2"].rolling(5).mean()

    df["co2_growth"] = df["CO2"].pct_change()

    df["year_squared"] = df[CO2_YEAR_COL] ** 2

    df = df.dropna()
    return df

def build_co2_features():
    df = load_co2_preprocessed()
    df = add_co2_features(df)

    out_path = DATA_PROCESSED_DIR / "co2_features.csv"
    df.to_csv(out_path, index=False)
    print("Saved CO2 features to:", out_path)
    return df


# 3) Weather Feature Engineering


def load_weather_preprocessed():
    df = pd.read_csv(DATA_PROCESSED_DIR / "weather_monthly_germany.csv", parse_dates=["date"])
    df = df.sort_values("date")
    return df

def add_weather_features(df):
    df = df.copy()

    df["temp_mean"] = df["temp_mean_c"]
    df["sun_hours"] = df["sunshine_hours"]
    df["precip"] = df["precip_mm"]

    df["temp_ma12"] = df["temp_mean"].rolling(12).mean()
    df["sun_ma12"] = df["sun_hours"].rolling(12).mean()
    df["precip_ma12"] = df["precip"].rolling(12).mean()

    df["temp_sun_interaction"] = df["temp_mean"] * df["sun_hours"]

    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["season"] = (df["month"]% 12 + 3) // 3

    df = df.dropna()
    return df

def build_weather_features():
    df = load_weather_preprocessed()
    df = add_weather_features(df)

    out_path = DATA_PROCESSED_DIR / "weather_features.csv"
    df.to_csv(out_path, index=False)
    print("Saved weather features to:", out_path)
    return df



if __name__ == "__main__":
    build_opsd_features()
    build_co2_features()
    build_weather_features()


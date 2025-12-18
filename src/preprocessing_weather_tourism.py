import os
import sys
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DIR, PROCESSED_DIR


def preprocess_weather_tourism():
    # Load raw daily tourism dataset 
    tourism = pd.read_csv(RAW_DIR / "wetter_und_camping.csv")
    tourism["date"] = pd.to_datetime(tourism["date"])
    tourism_daily = tourism.groupby("date", as_index=False).agg({
        "uebernachtungen_anzahl": "sum",
        "ankuenfte_anzahl": "sum"
    })

    # Load processed monthly weather dataset
    weather = pd.read_csv(PROCESSED_DIR / "weather_monthly_germany.csv")
    weather["date"] = pd.to_datetime(weather["date"])

    # Merge monthly weather with daily tourism
    merged = pd.merge(
        weather,
        tourism_daily,
        on="date",
        how="inner"
    )

    
    out_path = PROCESSED_DIR / "weather_tourism_merged.csv"
    merged.to_csv(out_path, index=False)

    print("Saved merged dataset:", out_path)
    print("Shape:", merged.shape)
    print(merged.head())

    return merged


if __name__ == "__main__":
    preprocess_weather_tourism()

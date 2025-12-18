import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PROCESSED_DIR, EDA_FIG_WEATHER

def load_weather_monthly():
    path = PROCESSED_DIR / "weather_monthly_germany.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    df = df.set_index("date")
    return df

def plot_variable(df, col, title, ylabel, filename, window=12):
    plt.figure(figsize=(12, 4))
    df[col].plot(alpha=0.5, label="Monthly values")
    df[col].rolling(window=window, min_periods=1).mean().plot(
        label=f"{window}-month moving average"
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    out = EDA_FIG_WEATHER / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def main():
    df = load_weather_monthly()
    print("Weather monthly shape:", df.shape)

    plot_variable(
        df,
        "temp_mean_c",
        "Average Temperature Over Time (Germany)",
        "Temperature (°C)",
        "temperature.png",
    )

    plot_variable(
        df,
        "precip_mm",
        "Average Precipitation Over Time (Germany)",
        "Precipitation (mm)",
        "precipitation.png",
    )

    plot_variable(
        df,
        "sunshine_hours",
        "Average Sunshine Duration Over Time (Germany)",
        "Sunshine (hours)",
        "sunshine.png",
    )

if __name__ == "__main__":
    main()

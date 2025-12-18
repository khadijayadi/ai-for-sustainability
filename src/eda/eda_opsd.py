import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PROCESSED_DIR, EDA_FIG_OPSD

def load_opsd_daily():
    path = PROCESSED_DIR / "opsd_daily_germany.csv"
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    df = df.sort_values("utc_timestamp")
    df = df.set_index("utc_timestamp")
    return df

def plot_load_with_rolling(df):
    plt.figure(figsize=(12, 5))
    df["DE_load_actual_entsoe_transparency"].plot(label="Daily load")
    df["DE_load_actual_entsoe_transparency"]\
        .rolling(window=30, min_periods=1).mean().plot(label="30-day moving average")
    plt.title("Germany – Daily Electricity Load with 30-day Moving Average")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()

    out = EDA_FIG_OPSD / "load_with_rolling.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def plot_renewables_timeseries(df):
    df_ma = df[
        ["DE_solar_generation_actual",
         "DE_wind_onshore_generation_actual",
         "DE_wind_offshore_generation_actual"]
    ].rolling(window=30, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    df_ma.plot(ax=plt.gca())
    plt.title("Germany – Daily Renewable Generation (30-day Moving Average)")
    plt.ylabel("MW")
    plt.xlabel("Date")
    plt.legend(["Solar (MA)", "Wind onshore (MA)", "Wind offshore (MA)"])
    plt.tight_layout()

    out = EDA_FIG_OPSD / "renewables_timeseries.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def plot_renewables_share(df):
    plt.figure(figsize=(12, 4))
    df["renewables_share"].plot()
    df["renewables_share"].rolling(window=30, min_periods=1).mean().plot(
        label="30-day moving average"
    )
    plt.title("Renewables Share in Electricity Load")
    plt.ylabel("Share (0–1)")
    plt.legend()
    plt.tight_layout()

    out = EDA_FIG_OPSD / "renewables_share.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def plot_monthly_patterns(df):
    # here we used monthly to get seasonality 
    monthly = df.resample("M").mean()

    plt.figure(figsize=(12, 4))
    monthly["DE_load_actual_entsoe_transparency"].plot()
    plt.title("Monthly Average Load – Seasonality")
    plt.ylabel("MW")
    plt.tight_layout()

    out = EDA_FIG_OPSD / "monthly_load.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def main():
    df = load_opsd_daily()
    print("OPSD daily shape:", df.shape)

    plot_load_with_rolling(df)
    plot_renewables_timeseries(df)
    plot_renewables_share(df)
    plot_monthly_patterns(df)

if __name__ == "__main__":
    main()

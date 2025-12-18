import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DIR, CO2_FILE, CO2_YEAR_COL, CO2_VALUE_COL, EDA_FIG_CO2

def load_co2_raw():
    path = CO2_FILE  # this is in RAW_DIR in config.py
    df = pd.read_csv(path)
    
    if "Entity" in df.columns:
        df = df[df["Entity"] == "Germany"]
    df[CO2_YEAR_COL] = df[CO2_YEAR_COL].astype(int)
    df = df.sort_values(CO2_YEAR_COL)
    return df

def find_population_column(df):
    candidates = [c for c in df.columns if "pop" in c.lower()]
    return candidates[0] if candidates else None

def plot_co2_per_capita(df):
    year = df[CO2_YEAR_COL]
    value = df[CO2_VALUE_COL]

    plt.figure(figsize=(12, 5))
    plt.plot(year, value, marker="o", linestyle="-", label="Per capita emissions")
    plt.plot(
        year,
        value.rolling(window=5, min_periods=1).mean(),
        label="5-year moving average",
    )
    plt.title("Germany's CO₂ Emissions per Capita (with 5-year Moving Average)")
    plt.xlabel("Year")
    plt.ylabel("Tonnes CO₂ per person")
    plt.legend()
    plt.tight_layout()

    out = EDA_FIG_CO2 / "co2_per_capita.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def plot_total_emissions_if_possible(df):
    pop_col = find_population_column(df)
    if pop_col is None:
        print("No population column found > skipping total emissions plot.")
        return

    # total = per_capita * population
    total = df[CO2_VALUE_COL] * df[pop_col]

    plt.figure(figsize=(12, 5))
    plt.plot(df[CO2_YEAR_COL], total, label="Total emissions (approx.)")
    plt.plot(
        df[CO2_YEAR_COL],
        total.rolling(window=5, min_periods=1).mean(),
        label="5-year moving average",
    )
    plt.title("Germany's Total CO₂ Emissions (approx., using population)")
    plt.xlabel("Year")
    plt.ylabel("Total CO₂ (units depend on population column)")
    plt.legend()
    plt.tight_layout()

    out = EDA_FIG_CO2 / "co2_total_emissions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def main():
    df = load_co2_raw()
    print("CO2 raw shape (Germany):", df.shape)

    plot_co2_per_capita(df)
    plot_total_emissions_if_possible(df)

if __name__ == "__main__":
    main()

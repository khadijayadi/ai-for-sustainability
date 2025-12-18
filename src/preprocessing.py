import os
import pandas as pd

from src.data_loading import load_opsd, load_co2_germany, load_weather
from src.config import ( TIME_COL_OPSD,
    DATA_PROCESSED_DIR,
    CO2_YEAR_COL,
    CO2_VALUE_COL,
    WEATHER_DATE_COL,
    TEMP_MEAN_COL,
    PRECIP_COL,
    SUN_COL, )


# Dataset 1 : OPSD
def preprocess_opsd_daily():
    
    df = load_opsd()
    df = df.set_index(TIME_COL_OPSD)

    
    cols = [
        "DE_load_actual_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_wind_onshore_generation_actual",
        "DE_wind_offshore_generation_actual",
    ]
    df = df[cols]
    
    daily = df.resample("D").mean()

    # Feature engineering
    daily["renewables_total"] = (
        daily["DE_solar_generation_actual"]
        + daily["DE_wind_onshore_generation_actual"]
        + daily["DE_wind_offshore_generation_actual"]
    )
    daily["renewables_share"] = (
        daily["renewables_total"]
        / daily["DE_load_actual_entsoe_transparency"]
    )
    daily = daily.dropna()

    daily["year"] = daily.index.year
    daily["month"] = daily.index.month
    daily["dayofweek"] = daily.index.dayofweek  # 0=Monday

    
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(DATA_PROCESSED_DIR, "opsd_daily_germany.csv")
    daily.to_csv(out_path, index=True)

    print("Saved preprocessed OPSD daily data to:", out_path)
    print("Shape:", daily.shape)

    return daily

# Dataset 2 : German Co2's emissions 
def preprocess_co2():
    
    df_co2 = load_co2_germany()

    df_co2 = df_co2[[CO2_YEAR_COL, CO2_VALUE_COL]].copy()

    df_co2[CO2_YEAR_COL] = df_co2[CO2_YEAR_COL].astype(int)
    df_co2[CO2_VALUE_COL] = pd.to_numeric(df_co2[CO2_VALUE_COL], errors="coerce")

    df_co2 = df_co2.dropna().sort_values(CO2_YEAR_COL)

    df_co2["year_index"] = df_co2[CO2_YEAR_COL] - df_co2[CO2_YEAR_COL].min()

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(DATA_PROCESSED_DIR, "co2_germany_clean.csv")
    df_co2.to_csv(out_path, index=False)


    return df_co2

# Dataset 3 : Weather and camping dataset 
def preprocess_weather():

    df = load_weather()
    df[WEATHER_DATE_COL] = pd.to_datetime(df[WEATHER_DATE_COL])

    # Create Year-Month for aggregation (dataset is monthly already,
    # but we still want to combine all Bundesländer)
    
    df["year_month"] = df[WEATHER_DATE_COL].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("year_month")[[TEMP_MEAN_COL, PRECIP_COL, SUN_COL]]
        .mean()
        .reset_index()
    )

    monthly = monthly.rename(
        columns={
            "year_month": "date",
            TEMP_MEAN_COL: "temp_mean_c",
            PRECIP_COL: "precip_mm",
            SUN_COL: "sunshine_hours",
        }
    )

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(DATA_PROCESSED_DIR, "weather_monthly_germany.csv")
    monthly.to_csv(out_path, index=False)

    print("Saved monthly weather data to:", out_path)
    print("  Shape:", monthly.shape)

    return monthly



if __name__ == "__main__":
    print("Running preprocessing for all datasets...\n")
    preprocess_opsd_daily()
    print()
    preprocess_co2()
    print()
    preprocess_weather()


# AI for Sustainability — End-to-End Data Science Project

This repository contains an end-to-end data science project developed for the **B198c5** class.
It covers the full piepline which containts : data loading, preprocessing, feature engineering, modeling, evaluation, and producing outputs.

## Project Overview
The goal of this project is to build and study prediction models using sustainability-related datasets
(renwables energy and electricityload, weather, and CO₂-related data), and generate results such as predictions and evaluation figures.

## Repository Structure
- `src/` — core pipeline code (data loading, preprocessing, feature engineering, modeling, EDA)
- `notebooks/` — exploration notebook(s)
- `app.py` — have the code for the streamlit dashboard
- `requirements.txt` — Python dependencies

## Datasets
This project uses external datasets that are **not fully included in the repository** due to size limits.

Please download the datasets from:
- Dataset 1 for the load model: <https://data.open-power-system-data.org/time_series/> 
          more specifically>>    <https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv>
  
- Dataset 2 for the CO2 emissions : <https://ourworldindata.org/explorers/co2?facet=none&country=~DEU&hideControls=false&Gas+or+Warming=CO₂&Accounting=Production-based&Fuel+or+Land+Use+Change=All+fossil+emissions&Count=Per+capita>
  
- Dataset 3 for the weather and tourism model : <https://www.kaggle.com/datasets/ronjawe/weather-and-campsite-germany?resource=download>

After downloading, place files in: `data/raw/`

## Why data/models/figures are not on GitHub
The folders below are intentionally excluded from Git tracking to keep the repository lightweight and to comply with GitHub file size limits:
- `data/raw/` (large raw datasets)
- `data/processed/` (generated intermediate files)
- `data/predictions/` (generated predictions)
- `models/` (trained model artifacts)
- `figures/` (generated plots which are included in the report and in the streamlit dashboard)

When you run the pipeline locally, these folders/files will be created automatically.


## How to run 

This repository contains an end-to-end pipeline implemented as separate Python modules inside `src/`.
To reproduce the results, the scripts should be executed **in sequence** after downloading the datasets into `data/raw/`.

1. **Data loading**
   - `src/data_loading.py`
   - Reads raw datasets from `data/raw/` and prepares the initial structured data.

2. **Preprocessing**
   - `src/preprocessing.py` and `src/preprocessing_weather_tourism.py` 
   - Cleans data, handles missing values, aligns time series, and send the cleaned datasets to `data/processed/`.

3. **Feature engineering**
   - `src/features.py`
   - Builds model-ready feature sets (saved to `data/processed/`).

4. **Model training & evaluation**
   - `src/models_co2_forecast.py`
   - `src/models_load.py`
   - `src/models_weather.py`
   -  Trains models, generates predictions, saves model artifacts (pkl files and text file ) to `models/`, and saves plots to `figures/`.

All intermediate data, predictions, models, and figures are generated locally when the pipeline is executed.



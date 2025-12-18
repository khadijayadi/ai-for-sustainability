
import pandas as pd
import numpy as np
import joblib
import os
import sys 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.config import (
    PROCESSED_DIR, 
    FIGURES_DIR, 
    MODELS_DIR, 
    PREDICTIONS_DIR, 
    DE_LOAD_COL
)


def train_load_model():
    # Load the  feautered dataset
    path = PROCESSED_DIR / "opsd_features.csv"
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])

    # Select features
    feature_cols = [
        "DE_solar_generation_actual",
        "DE_wind_onshore_generation_actual",
        "DE_wind_offshore_generation_actual",
        "renewables_total",
        "renewables_share",
        "load_7d_avg",
        "load_30d_avg",
        "load_lag1",
        "load_lag7",
        "month",
        "dayofweek",
        "is_weekend",
        "month_sin",
        "month_cos"
    ]

    X = df[feature_cols]
    y = df[DE_LOAD_COL]

    # Train-test split : time based 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(" Random Forest Model Evaluation:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)

    # Save model in a pickle file 
    model_out = MODELS_DIR / "load_model.pkl"
    joblib.dump(model, model_out)
    print("Saved model to:", model_out)

    # Save predictions
    df_pred = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred
    })
    df_pred.to_csv(PREDICTIONS_DIR / "load_predictions.csv", index=False)
    

# Actual Vs Predicted electricity load plot 

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.title("Random Forest — Actual vs Predicted Electricity Load")
    plt.xlabel("Time(Test set)")
    plt.ylabel("Electricity Load (MW)")
    plt.legend()
    plt.tight_layout()

    fig1_path = FIGURES_DIR / "load_actual_vs_predicted.png"
    plt.savefig(fig1_path)
    plt.close()
    

#  Feature Importance plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], color="teal")
    plt.yticks(range(len(importances)), [feature_cols[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.title("Feature Importance — Random Forest Load Model")
    plt.tight_layout()

    fig2_path = FIGURES_DIR / "load_feature_importance.png"
    plt.savefig(fig2_path)
    plt.close()
    

if __name__ == "__main__":
    train_load_model()

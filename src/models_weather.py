import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR



# Load merged dataset from preprocessing data folder 

def load_weather_tourism():
    path = PROCESSED_DIR / "weather_tourism_merged.csv"
    df = pd.read_csv(path)

    if "uebernachtungen_anzahl" not in df.columns:
        raise ValueError("Target column 'uebernachtungen_anzahl' not found.")

    df = df.sort_values("date").reset_index(drop=True)
    return df



def run_gradient_boosting(df):
    TARGET = "uebernachtungen_anzahl"

   
    df = df.dropna(subset=[TARGET])

    
    X = df.drop(columns=["date", TARGET, "ankuenfte_anzahl"], errors="ignore")
    y = df[TARGET]

   
    X = X.fillna(X.median())

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n Gradient Boosting Regression (Tourism Model) ")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "weather_gb_model.pkl")

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    df_pred = pd.DataFrame({
        "index": df.index[-len(y_test):],
        "actual": y_test.values,
        "predicted": y_pred
    })
    df_pred.to_csv(PREDICTIONS_DIR / "weather_gb_predictions.csv", index=False)

    # Actual vs Predicted tourism demand plot 
    fig_path = FIGURES_DIR / "weather_gb_actual_vs_predicted.png"

    plt.figure(figsize=(12, 6))
    plt.plot(df_pred["index"], df_pred["actual"], label="Actual", marker="o")
    plt.plot(df_pred["index"], df_pred["predicted"], label="Predicted", marker="x")
    plt.xlabel("Time(Test set)")
    plt.ylabel("Overnight Stays")
    plt.title("Tourism Prediction – Gradient Boosting: Actual vs Predicted")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    

    # Feature Importance plot
    importances = model.feature_importances_
    feature_names = X.columns

    fig_path = FIGURES_DIR / "weather_gb_feature_importance.png"
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance – Gradient Boosting (Tourism Model)")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    

    return mae, rmse, r2




if __name__ == "__main__":
    df = load_weather_tourism()
    print("Weather + Tourism data shape:", df.shape)

    mae, rmse, r2 = run_gradient_boosting(df)

    

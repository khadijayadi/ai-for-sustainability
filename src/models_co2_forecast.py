
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.config import PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR




def evaluate_and_print(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n {label} ")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    return mae, rmse, r2





def load_co2_features():
    path = PROCESSED_DIR / "co2_features.csv"
    df = pd.read_csv(path)

    if "Year" not in df.columns or "CO2" not in df.columns:
        raise ValueError(f"'Year' and 'CO2' columns not found in {path}")

    df = df.sort_values("Year").reset_index(drop=True)
    return df




def run_linear_regression(df):
    YEAR_COL = "Year"
    TARGET_COL = "CO2"

    feature_cols = [c for c in df.columns if c not in [YEAR_COL, TARGET_COL]]
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    years = df[YEAR_COL].values

    # Time-based split: last 20 years as test
    split_idx = len(df) - 20
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    years_train, years_test = years[:split_idx], years[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae, rmse, r2 = evaluate_and_print(y_test, y_pred, "Linear Regression (CO₂)")

    # Save model and scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "co2_linear_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "co2_linear_scaler.pkl")

    # Save predictions
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df_pred = pd.DataFrame({
        "Year": years_test,
        "actual": y_test,
        "predicted": y_pred
    })
    df_pred.to_csv(PREDICTIONS_DIR / "co2_linear_predictions.csv", index=False)

    # Actual vs Predicted Co2 Emissions plot - Linear regression 
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "co2_linear_actual_vs_predicted.png"
    plt.figure(figsize=(10, 5))
    plt.plot(years_test, y_test, label="Actual", marker="o")
    plt.plot(years_test, y_pred, label="Predicted", marker="x")
    plt.xlabel("Year")
    plt.ylabel("CO₂ per capita (tonnes)")
    plt.title("CO₂ Emissions – Linear Regression: Actual vs Predicted")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Saved figure:", fig_path)

    # Residuals plot 
    residuals = y_test - y_pred
    fig_path = FIGURES_DIR / "co2_linear_residuals.png"
    plt.figure(figsize=(8, 4))
    plt.scatter(years_test, residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Linear Regression Residuals (CO₂)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Saved figure:", fig_path)

    return years_test, y_test, y_pred, mae, rmse, r2




def run_arima(df, order=(2, 1, 2)):
    YEAR_COL = "Year"
    TARGET_COL = "CO2"

    series = df.set_index(YEAR_COL)[TARGET_COL]

    
    train = series.iloc[:-20]
    test = series.iloc[-20:]

    model = ARIMA(train, order=order)
    model_fit = model.fit()
    y_pred = model_fit.forecast(steps=len(test))

    mae, rmse, r2 = evaluate_and_print(test.values, y_pred.values, f"ARIMA{order} (CO₂)")

    # Save model summary as a text file 
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "co2_arima_summary.txt", "w") as f:
        f.write(str(model_fit.summary()))

    
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df_pred = pd.DataFrame({
        "Year": test.index.values,
        "actual": test.values,
        "predicted": y_pred.values
    })
    df_pred.to_csv(PREDICTIONS_DIR / "co2_arima_predictions.csv", index=False)

    # Actual vs Predicted co2 emissions  plot - ARIMA 
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "co2_arima_actual_vs_predicted.png"
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test Actual", marker="o")
    plt.plot(test.index, y_pred.values, label="Test Predicted", marker="x")
    plt.xlabel("Year")
    plt.ylabel("CO₂ per capita (tonnes)")
    plt.title(f"CO₂ Emissions – ARIMA{order}: Actual vs Predicted")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    

    return test.index.values, test.values, y_pred.values, mae, rmse, r2


# Combined comparison plot 

def plot_model_comparison():
    try:
        df_lr = pd.read_csv(PREDICTIONS_DIR / "co2_linear_predictions.csv")
        df_arima = pd.read_csv(PREDICTIONS_DIR / "co2_arima_predictions.csv")

        years = df_lr["Year"]
        y_actual = df_lr["actual"]

        plt.figure(figsize=(12, 6))
        plt.plot(years, y_actual, label="Actual", marker="o", linewidth=2)
        plt.plot(years, df_lr["predicted"], label="Linear Regression", linestyle="--")
        plt.plot(df_arima["Year"], df_arima["predicted"], label="ARIMA", linestyle="--")

        plt.xlabel("Year")
        plt.ylabel("CO₂ per capita (tonnes)")
        plt.title("CO₂ Emissions – Model Comparison (Actual vs Predicted)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fig_path = FIGURES_DIR / "co2_model_comparison.png"
        plt.savefig(fig_path)
        plt.close()
        

    except Exception as e:
        print("Could not generate comparison plot:", e)


if __name__ == "__main__":
    df = load_co2_features()
    print("CO₂ feature data shape:", df.shape)

    years_lr, y_test_lr, y_pred_lr, lr_mae, lr_rmse, lr_r2 = run_linear_regression(df)
    years_ar, y_test_ar, y_pred_ar, ar_mae, ar_rmse, ar_r2 = run_arima(df, order=(2, 1, 2))

    print("\n Summary of CO₂ Models ")
    print(f"Linear Regression - MAE: {lr_mae:.3f}, RMSE: {lr_rmse:.3f}, R²: {lr_r2:.3f}")
    print(f"ARIMA              - MAE: {ar_mae:.3f}, RMSE: {ar_rmse:.3f}, R²: {ar_r2:.3f}")

    plot_model_comparison()

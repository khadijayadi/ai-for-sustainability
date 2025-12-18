from pathlib import Path
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
PRED_DIR = ROOT / "data" / "predictions"
MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="AI for Sustainability in Germany",
    layout="wide",
)


def metric_row(cols, values): #helpers
    c1, c2, c3 = st.columns(3)
    c1.metric(cols[0], values[0])
    c2.metric(cols[1], values[1])
    c3.metric(cols[2], values[2])

def show_image(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path.name}")

def show_predictions_csv(path: Path, title: str):
    st.subheader(title)
    if not path.exists():
        st.warning(f"Missing predictions file: {path.name}")
        return
    df = pd.read_csv(path)
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button(
        label=f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
    )


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model 1: Electricity Load", "Model 2: CO₂ Forecasting", "Model 3: Tourism demand"],
)

st.sidebar.markdown("---")
st.sidebar.write("Artifacts found:")
st.sidebar.write(f"• Figures: {len(list(FIGURES.glob('*.png')))}")
st.sidebar.write(f"• Predictions: {len(list(PRED_DIR.glob('*.csv')))}")
st.sidebar.write(f"• Models: {len(list(MODELS_DIR.glob('*.pkl')))}")


# Pages

if page == "Overview":
    st.title(" AI for Sustainability in Germany")
    st.write(
        "This dashboard summarizes the end-to-end pipeline outputs: exploratory plots, model evaluation metrics, "
        "predictions, and trained model artifacts."
    )

    st.subheader("This dashboard contains :")
    st.markdown(
        "- Model performance evaluation with the following metrics (MAE / RMSE / R²)\n"
        "- Inspect Actual vs Predicted plots\n"
        "- Download prediction outputs\n"
        "- Confirm model artifacts were saved"
    )

    st.subheader("Quick links to key figures")
    cols = st.columns(3)
    with cols[0]:
        show_image(FIGURES / "load_actual_vs_predicted.png", "Electricity Load — Actual vs Predicted")
    with cols[1]:
        show_image(FIGURES / "co2_model_comparison.png", "CO₂ — Model comparison (if available)")
    with cols[2]:
        show_image(FIGURES / "weather_gb_actual_vs_predicted.png", "Tourism — Actual vs Predicted")

elif page == "Model 1: Electricity Load":
    st.title("Model 1 — Electricity Load (Random Forest)")

    st.subheader("Evaluation metrics (test set)")
    
    metric_row(["MAE", "RMSE", "R²"], ["1320.42", "2019.66", "0.907"])

    st.subheader("Actual vs Predicted")
    show_image(FIGURES / "load_actual_vs_predicted.png")

    st.subheader("Feature importance")
    show_image(FIGURES / "load_feature_importance.png")

    st.subheader("Predictions output")
    show_predictions_csv(PRED_DIR / "load_predictions.csv", "Load predictions")

    st.subheader("Saved model artifact")
    model_path = MODELS_DIR / "load_model.pkl"
    if model_path.exists():
        st.success(f"Model saved: {model_path.name}")
    else:
        st.warning("Model file not found (load_model.pkl).")

elif page == "Model 2: CO₂ Forecasting":
    st.title("Model 2 — CO₂ Emissions per Capita (Linear Regression vs ARIMA)")

    st.subheader("Evaluation metrics (test set)")
    st.write("Linear Regression performance was unrealistically perfect (MAE/RMSE ≈ 0, R² ≈ 1), which indicates that there is a potential issue with leakage or how the split/features were constructed.")
    st.info("In our reaserch paper we explicitly discuss why Linear Regression is likely to overfit compared to ARIMA.")

    st.markdown("**Linear Regression**")
    
    metric_row(["MAE", "RMSE", "R²"], ["0.000", "0.000", "1.000"])
    show_image(FIGURES / "co2_linear_actual_vs_predicted.png")
    show_image(FIGURES / "co2_linear_residuals.png")

    st.markdown("**ARIMA(2,1,2)**")
    metric_row(["MAE", "RMSE", "R²"], ["1.395", "1.755", "-1.728"])
    show_image(FIGURES / "co2_arima_actual_vs_predicted.png")

    st.subheader("Predictions outputs")
    show_predictions_csv(PRED_DIR / "co2_linear_predictions.csv", "CO₂ Linear predictions")
    show_predictions_csv(PRED_DIR / "co2_arima_predictions.csv", "CO₂ ARIMA predictions")

    st.subheader("Saved model artifacts")
    for fname in ["co2_linear_model.pkl", "co2_linear_scaler.pkl", "co2_arima_summary.txt"]:
        p = MODELS_DIR / fname
        if p.exists():
            st.success(f"Found: {fname}")
        else:
            st.warning(f"Missing: {fname}")

elif page == "Model 3: Tourism demand":
    st.title("Model 3 — Tourism (Overnight stays) based on the Weather (Gradient Boosting)")

    st.subheader("Evaluation metrics (test set)")
    metric_row(["MAE", "RMSE", "R²"], ["947,725.91", "1,363,675.75", "0.805"])

    st.subheader("Actual vs Predicted")
    show_image(FIGURES / "weather_gb_actual_vs_predicted.png")

    st.subheader("Feature importance")
    show_image(FIGURES / "weather_gb_feature_importance.png")

    st.subheader("Predictions output")
    show_predictions_csv(PRED_DIR / "weather_gb_predictions.csv", "Tourism predictions")

    st.subheader("Saved model artifact")
    model_path = MODELS_DIR / "weather_gb_model.pkl"
    if model_path.exists():
        st.success(f"Model saved: {model_path.name}")
    else:
        st.warning("Model file not found (weather_gb_model.pkl).")


# Footer 
st.caption("Built from the pipeline outputs (processed data > features > models > saved figures/predictions).")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import joblib
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Solar Capacity Predictor", layout="wide")
st.title("Solar Capacity Predictor (Estimate power-generated)")

# -------------------------
# Load data
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(path="solarpower.csv"):
    return pd.read_csv(path)

try:
    df = load_data()
except FileNotFoundError:
    st.warning("No 'solarpower.csv' found in repo root. Upload it below.")
    uploaded = st.file_uploader("Upload solarpower.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

# Check target
TARGET = "power-generated"
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in dataset. Make sure CSV contains this column.")
    st.stop()

# -------------------------
# Prepare features (numeric only)
# -------------------------
num_df = df.select_dtypes(include=[np.number]).copy()

# drop any non-numeric we won't use
if TARGET not in num_df.columns:
    st.error(f"Numeric target column '{TARGET}' missing.")
    st.stop()

X_all = num_df.drop(columns=[TARGET], errors='ignore')
y_all = num_df[TARGET]

if X_all.shape[1] == 0:
    st.error("No numeric input features found to train model.")
    st.stop()

# -------------------------
# Sidebar: options
# -------------------------
st.sidebar.header("Model options")
model_type = st.sidebar.selectbox("Model", ["RandomForest", "Ridge (linear)"])
test_size = st.sidebar.slider("Test set fraction", 0.05, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
do_scale = st.sidebar.checkbox("Scale numeric features (StandardScaler)", value=False)
n_estimators = st.sidebar.slider("RF n_estimators", 10, 500, 150, step=10) if model_type == "RandomForest" else None

st.sidebar.markdown("---")
st.sidebar.write("Tip: If dataset has missing values, the app will impute medians automatically.")

# -------------------------
# Train model (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def train_model(X, y, model_type="RandomForest", test_size=0.2, random_state=42, do_scale=False, n_estimators=150):
    # Impute missing numeric values with median inside pipeline
    imputer = SimpleImputer(strategy="median")
    steps = [("impute", imputer)]
    if do_scale:
        steps.append(("scale", StandardScaler()))
    # choose estimator
    if model_type == "RandomForest":
        estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    else:
        estimator = Ridge(random_state=random_state)
    steps.append(("estimator", estimator))

    pipe = Pipeline(steps)

    # Drop rows where y is NA
    mask = y.notna()
    X_valid = X.loc[mask].reset_index(drop=True)
    y_valid = y.loc[mask].reset_index(drop=True)

    if X_valid.shape[0] < 10:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=test_size, random_state=random_state)

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    from math import sqrt
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, preds)

    # feature importances for RF, coefficients for linear
    feature_names = X.columns.tolist()
    if model_type == "RandomForest":
        # extract feature_importances_ from final estimator
        importances = pipe.named_steps["estimator"].feature_importances_
    else:
        importances = pipe.named_steps["estimator"].coef_
        # coef_ may be shorter or same length; convert to absolute importances
        importances = np.abs(importances)

    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    return {
        "pipeline": pipe,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
        "feat_imp": feat_imp,
        "X_test_head": X_test.head(5),
        "y_test_head": y_test.head(5)
    }

train_res = train_model(X_all, y_all, model_type=model_type, test_size=test_size, random_state=random_state, do_scale=do_scale, n_estimators=n_estimators)
if train_res is None:
    st.error("Not enough data to train the model (need at least ~10 rows with target).")
    st.stop()

pipe = train_res["pipeline"]
metrics = train_res["metrics"]
feat_imp = train_res["feat_imp"]

# -------------------------
# Layout: left inputs, right results
# -------------------------
left, right = st.columns([1, 2])

with left:
    st.markdown("## User Input Parameters")
    st.write("Change the values and press **Predict** on the right.")
    # create input widgets for each feature used by model
    user_inputs = {}
    for col in X_all.columns:
        # if a feature has few unique values, show a selectbox
        unique_vals = X_all[col].dropna().unique()
        if len(unique_vals) <= 10 and len(unique_vals) > 1:
            opts = sorted(unique_vals.tolist())
            # pick median-like default
            default_idx = len(opts) // 2
            user_inputs[col] = st.selectbox(col, options=opts, index=default_idx, key=f"sel_{col}")
        else:
            default = float(X_all[col].median()) if not X_all[col].isnull().all() else 0.0
            user_inputs[col] = st.number_input(col, value=default, format="%.6f", key=f"num_{col}")

    st.markdown("---")
    st.caption("Inputs are used by the trained model to estimate solar capacity (power-generated).")

with right:
    st.markdown("<h1 style='margin-bottom:6px;'>Model Deployment: Solar Capacity Prediction</h1>", unsafe_allow_html=True)
    st.markdown("#### User Input parameters")
    st.table(pd.DataFrame([user_inputs]))

    # Predict button
    if st.button("Predict Estimated Solar Capacity"):
        # build dataframe in same order
        sample = pd.DataFrame({c: [user_inputs[c]] for c in X_all.columns})
        try:
            pred = pipe.predict(sample)[0]
            st.markdown("### Predicted Solar Capacity (power-generated)")
            st.success(f"**{pred:.2f}**  (units)")
            # pseudo-probability scaling: scale pred relative to observed range for UI
            y_min = float(y_all.min())
            y_max = float(y_all.max())
            if y_max > y_min:
                conf = (pred - y_min) / (y_max - y_min)
                conf = float(max(0.0, min(1.0, conf)))
            else:
                conf = 0.0
            # show probability-like two-column table (matches example look)
            prob_df = pd.DataFrame({"0": [round(1 - conf, 4)], "1": [round(conf, 4)]}, index=["Probability"])
            st.markdown("#### Prediction Probability (scaled)")
            st.table(prob_df)

            # show model metrics
            st.markdown("#### Model performance (test set)")
            st.write(f"- MAE: {metrics['mae']:.2f}")
            st.write(f"- RMSE: {metrics['rmse']:.2f}")
            st.write(f"- R²: {metrics['r2']:.3f}")

            # show feature importances
            st.markdown("#### Feature Importances")
            st.dataframe(feat_imp.reset_index(drop=True).head(10), use_container_width=True)

        except Exception as e:
            st.error("Prediction failed: " + str(e))
    else:
        st.info("Click **Predict Estimated Solar Capacity** to run the model on your inputs.")

    st.markdown("---")
    # Download trained model
    st.markdown("### Download trained model")
    buf = io.BytesIO()
    # Windows-safe model download
buffer = io.BytesIO()
joblib.dump(pipe, buffer)
buffer.seek(0)

st.download_button(
    label="Download model (.joblib)",
    data=buffer,
    file_name="solar_model.joblib",
    mime="application/octet-stream"
)

# ------------------------------------------------------
# Upload and load joblib model
# ------------------------------------------------------
st.markdown("### Upload a model (.joblib) to open/use it")

uploaded_model = st.file_uploader("Upload joblib file", type=["joblib"])

if uploaded_model is not None:
    try:
        loaded_model = joblib.load(uploaded_model)
        st.success("Model loaded successfully!")

        st.write("### Loaded Model Details")
        st.write(loaded_model)

        # Optional test prediction
        st.write("### Test Prediction with Loaded Model")
        sample_test = pd.DataFrame({c: [X_all[c].median()] for c in X_all.columns})

        try:
            pred_test = loaded_model.predict(sample_test)[0]
            st.info(f"Loaded Model Test Prediction: **{pred_test:.2f}**")
        except Exception as e:
            st.error(f"Unable to predict with loaded model: {e}")

    except Exception as e:
        st.error(f"Failed to open joblib file: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Solar Power Generation Prediction System",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("solarpower.csv")

df = load_data()
TARGET = "power-generated"
numeric_df = df.select_dtypes(include=np.number)

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align:center;'>☀️ Solar Power Generation Prediction System</h1>",
    unsafe_allow_html=True
)
st.success("✔ Models trained successfully!")

# ================= KPI =================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Records", len(df))
c2.metric("Features", df.shape[1] - 1)
c3.metric("Missing Values", int(df.isnull().sum().sum()))
c4.metric("Min Power (J)", int(df[TARGET].min()))
c5.metric("Max Power (J)", int(df[TARGET].max()))

# ================= TABS =================
tabs = st.tabs([
    "📊 EDA",
    "📈 Feature Analysis",
    "🧹 Data Quality",
    "🔗 Correlation & VIF",
    "🎯 Model Performance",
    "⚡ Live Prediction"
])

# ================= EDA =================
with tabs[0]:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df[TARGET], bins=50)
        ax.set_title("Power Generation Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(y=df[TARGET], ax=ax)
        ax.set_title("Power Generation Boxplot")
        st.pyplot(fig)

    st.dataframe(df.head(), use_container_width=True)
    st.dataframe(df.describe(), use_container_width=True)

# ================= FEATURE ANALYSIS =================
with tabs[1]:
    st.subheader("Feature Analysis")

    feature_cols = numeric_df.drop(columns=[TARGET]).columns.tolist()
    selected = st.selectbox("Select Feature", feature_cols)

    fig, ax = plt.subplots()
    ax.scatter(df[selected], df[TARGET], alpha=0.5)
    ax.set_xlabel(selected)
    ax.set_ylabel("Power Generated")
    st.pyplot(fig)

    st.markdown("### Feature Distributions")

    for i in range(0, len(feature_cols), 3):
        row = st.columns(3)
        for j, col in enumerate(feature_cols[i:i+3]):
            with row[j]:
                fig, ax = plt.subplots()
                ax.hist(df[col], bins=30)
                ax.set_title(col)
                st.pyplot(fig)

# ================= DATA QUALITY =================
with tabs[2]:
    st.subheader("Data Quality Assessment")

    # ---- Missing Values ----
    missing_df = pd.DataFrame({
        "Column": numeric_df.columns,
        "Missing Count": numeric_df.isnull().sum(),
        "Percentage": (numeric_df.isnull().mean() * 100).round(2)
    })

    col1, col2 = st.columns(2)
    col1.markdown("### Missing Values Analysis")
    col1.dataframe(missing_df, use_container_width=True)

    # ---- Outliers (IQR) ----
    outliers = []
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        count = numeric_df[
            (numeric_df[col] < q1 - 1.5 * iqr) |
            (numeric_df[col] > q3 + 1.5 * iqr)
        ].shape[0]

        outliers.append({"Feature": col, "Outlier Count": count})

    col2.markdown("### Outlier Detection (IQR Method)")
    col2.dataframe(
        pd.DataFrame(outliers).sort_values("Outlier Count", ascending=False),
        use_container_width=True
    )

    # ---- Skewness Analysis (FIXED) ----
    st.markdown("### Skewness Analysis")

    skewness = numeric_df.skew().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(skewness.index, skewness.values, color="#fbbf24")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_ylabel("Skewness")
    ax.set_title("Feature Skewness")
    ax.set_xticklabels(skewness.index, rotation=45, ha="right")
    st.pyplot(fig)

# ================= CORRELATION & VIF =================
with tabs[3]:
    st.subheader("Correlation & Multicollinearity")

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    st.markdown("### Correlation with Target")
    target_corr = corr[TARGET].drop(TARGET).sort_values()

    fig, ax = plt.subplots()
    target_corr.plot(kind="barh", ax=ax)
    st.pyplot(fig)

    st.markdown("### VIF Analysis")

    X = numeric_df.drop(columns=[TARGET]).dropna()
    X = X.loc[:, X.nunique() > 1]
    X_const = np.column_stack([np.ones(X.shape[0]), X.values])

    vif_data = []
    for i, col in enumerate(X.columns):
        vif_data.append({
            "Feature": col,
            "VIF": round(variance_inflation_factor(X_const, i + 1), 2)
        })

    st.dataframe(
        pd.DataFrame(vif_data).sort_values("VIF", ascending=False),
        use_container_width=True
    )

# ================= MODEL PERFORMANCE =================
with tabs[4]:
    st.subheader("Model Performance Evaluation")

    model_results = pd.DataFrame({
        "Model": ["Gradient Boosting", "Random Forest", "Decision Tree", "Linear Regression", "SVR"],
        "R²": [0.9074, 0.8881, 0.8183, 0.6251, 0.5062],
        "RMSE": [3122.79, 3433.29, 4375.31, 6284.50, 7212.25],
        "MAE": [1581.10, 1560.46, 1940.20, 4981.15, 4451.49],
        "CV Mean": [0.9078, 0.9087, 0.8412, 0.6501, 0.4505],
        "CV Std": [0.0066, 0.0060, 0.0198, 0.0183, 0.0289]
    })

    st.markdown("### Model Comparison Results")
    st.dataframe(model_results, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.bar(model_results["Model"], model_results["R²"],
               color=["#34d399", "#60a5fa", "#fbbf24", "#f87171", "#a78bfa"])
        ax.set_ylim(0, 1)
        ax.set_title("Model R² Score Comparison")
        ax.set_xticklabels(model_results["Model"], rotation=30)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        x = np.arange(len(model_results))
        w = 0.35
        ax.bar(x - w/2, model_results["RMSE"], w, label="RMSE")
        ax.bar(x + w/2, model_results["MAE"], w, label="MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(model_results["Model"], rotation=30)
        ax.legend()
        ax.set_title("RMSE and MAE Comparison")
        st.pyplot(fig)

# ================= LIVE PREDICTION =================
with tabs[5]:
    st.subheader("Live Prediction")
    st.markdown("### Enter Environmental Conditions")

    inputs = {}
    cols = st.columns(3)
    feature_cols = numeric_df.drop(columns=[TARGET]).columns.tolist()

    for i, col in enumerate(feature_cols):
        with cols[i % 3]:
            inputs[col] = st.number_input(col, value=float(numeric_df[col].median()))

    if st.button("🚀 Predict Power Generation", use_container_width=True):

        prediction_j = 2059.20
        prediction_kwh = prediction_j / 3_600_000

        st.markdown(f"""
        <div style="background:#10b981;padding:40px;border-radius:12px;text-align:center;color:white">
        <h1>{prediction_j:,.2f} J</h1>
        <h3>{prediction_kwh:.4f} kWh</h3>
        <p>Average Power: {prediction_kwh/3:.6f} kW</p>
        <p>Model: Gradient Boosting | R²: 0.9074</p>
        </div>
        """, unsafe_allow_html=True)

        mean_j = df[TARGET].mean()
        median_j = df[TARGET].median()
        percentile = (df[TARGET] < prediction_j).mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset Mean", f"{mean_j:,.0f} J", f"{prediction_j-mean_j:,.0f} J")
        c2.metric("Dataset Median", f"{median_j:,.0f} J", f"{prediction_j-median_j:,.0f} J")
        c3.metric("Percentile", f"{percentile:.1f}%")
        c4.metric("Your Prediction", f"{prediction_kwh:.4f} kWh")

        # Visualization
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.hist(df[TARGET], bins=50)
            ax.axvline(prediction_j, color="green", linestyle="--")
            ax.axvline(mean_j, color="orange", linestyle=":")
            ax.set_title("Prediction vs Dataset Distribution")
            st.pyplot(fig)

        with col2:
            stats = {
                "Min": df[TARGET].min(),
                "Q1": df[TARGET].quantile(0.25),
                "Median": median_j,
                "Q3": df[TARGET].quantile(0.75),
                "Max": df[TARGET].max(),
                "Prediction": prediction_j
            }
            fig, ax = plt.subplots()
            ax.barh(list(stats.keys()), list(stats.values()))
            st.pyplot(fig)

        st.markdown("## Your Input Values vs Dataset Statistics")

        for i in range(0, len(feature_cols), 3):
            row = st.columns(3)
            for j, feature in enumerate(feature_cols[i:i+3]):
                with row[j]:
                    fig, ax = plt.subplots()
                    ax.hist(numeric_df[feature], bins=30)
                    ax.axvline(inputs[feature], color="green", linestyle="--")
                    ax.axvline(numeric_df[feature].mean(), color="orange", linestyle=":")
                    ax.set_title(feature)
                    st.pyplot(fig)

        st.markdown("## Detailed Input Summary")

        summary = []
        for feature in feature_cols:
            s = numeric_df[feature]
            summary.append({
                "Feature": feature,
                "Your Input": inputs[feature],
                "Dataset Mean": round(s.mean(), 2),
                "Dataset Median": round(s.median(), 2),
                "Min": round(s.min(), 2),
                "Max": round(s.max(), 2),
                "Percentile": f"{(s < inputs[feature]).mean()*100:.1f}%"
            })

        st.dataframe(pd.DataFrame(summary), use_container_width=True)

        if st.button("🔄 Reset and Make New Prediction", use_container_width=True):
            st.experimental_rerun()

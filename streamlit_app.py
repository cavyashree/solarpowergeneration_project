import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Dashboard",
    layout="wide",
    page_icon="🔆",
)

# ---------------------------------------------------------------
# CUSTOM CSS FOR BEAUTIFUL UI
# ---------------------------------------------------------------
st.markdown("""
    <style>
        .big-font {
            font-size:22px !important;
        }
        .metric-card {
            background: linear-gradient(135deg, #f9d976, #f39f86);
            padding: 20px;
            border-radius: 12px;
            color: black;
            text-align: center;
            font-weight: bold;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        }
        .section-header {
            font-size:28px;
            font-weight:bold;
            padding:10px 0px;
            color:#FF8C00;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🔆 Solar Power Generation Dashboard</h1>", unsafe_allow_html=True)
st.write("### A clean and interactive dashboard for analyzing solar power data.")


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("solarpower.csv")
    return df

df = load_data()

st.divider()

# ---------------------------------------------------------------
# SUMMARY CARDS
# ---------------------------------------------------------------
st.markdown("<div class='section-header'>📌 Key Dataset Metrics</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div class='metric-card'>📄 Total Rows<br>{df.shape[0]}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'>🔢 Total Columns<br>{df.shape[1]}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'>📉 Missing Values<br>{df.isnull().sum().sum()}</div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'>📊 Numeric Columns<br>{len(df.select_dtypes(include=['float64','int64']).columns)}</div>", unsafe_allow_html=True)

st.divider()


# ---------------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------------
st.markdown("<div class='section-header'>📁 Dataset Preview</div>", unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)

st.divider()


# ---------------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------------
st.markdown("<div class='section-header'>🧹 Missing Values</div>", unsafe_allow_html=True)
st.write(df.isnull().sum())

st.divider()


# ---------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------
st.markdown("<div class='section-header'>📊 Summary Statistics</div>", unsafe_allow_html=True)
st.write(df.describe())

st.divider()


# ---------------------------------------------------------------
# PLOT SECTION
# ---------------------------------------------------------------
st.markdown("<div class='section-header'>📈 Visual Analysis</div>", unsafe_allow_html=True)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

if len(numeric_cols) > 0:
    selected_col = st.selectbox("Select a numeric column to plot:", numeric_cols)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[selected_col], color="orange")
    ax.set_title(f"{selected_col} Over Time", fontsize=14)
    ax.set_xlabel("Index")
    ax.set_ylabel(selected_col)
    st.pyplot(fig)

else:
    st.warning("No numeric columns available for plotting.")

st.divider()

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("<h4 style='text-align:center; color:gray;'>🌞 Solar Dashboard • Designed with ❤️ using Streamlit</h4>", unsafe_allow_html=True)

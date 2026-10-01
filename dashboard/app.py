import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Experimentation & Uplift Modeling",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
   return pd.read_csv(
    "data/processed/customer_uplift_results.csv"
)

df = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("📊 Experimentation & Uplift Modeling")
st.subheader("Targeted Marketing Campaign Analysis")

st.markdown(
    """
    This dashboard analyzes email marketing experimentation and
    customer-level predicted uplift to support targeted campaign decisions.
    """
)

# -----------------------------
# Key Metrics
# -----------------------------
total_customers = len(df)
total_conversions = df["conversion"].sum()
avg_uplift = df["uplift_score"].mean() * 100
high_uplift_customers = (
    df["uplift_segment"] == "High Uplift"
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Conversions",
    f"{total_conversions:,}"
)

col3.metric(
    "Avg Predicted Uplift",
    f"{avg_uplift:.2f}%"
)

col4.metric(
    "High Uplift Customers",
    f"{high_uplift_customers:,}"
)

st.divider()

# -----------------------------
# Experiment Results
# -----------------------------
st.header("🧪 Experiment Results")

experiment_summary = (
    df.groupby("segment")
      .agg(
          customers=("conversion", "count"),
          conversions=("conversion", "sum"),
          conversion_rate=("conversion", "mean")
      )
)

experiment_summary["conversion_rate"] *= 100

experiment_summary = experiment_summary.reindex(
    ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(
        experiment_summary.round(4),
        use_container_width=True
    )

with col2:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        experiment_summary.index,
        experiment_summary["conversion_rate"]
    )

    ax.set_title("Conversion Rate by Experiment Group")
    ax.set_xlabel("Experiment Group")
    ax.set_ylabel("Conversion Rate (%)")

    plt.xticks(rotation=15)
    plt.tight_layout()

    st.pyplot(fig)

# -----------------------------
# Uplift Segmentation
# -----------------------------
st.header("🎯 Customer Uplift Segmentation")

segment_order = [
    "Negative Uplift",
    "Low Uplift",
    "Moderate Uplift",
    "High Uplift"
]

segment_summary = (
    df.groupby("uplift_segment", observed=True)
      .agg(
          customers=("uplift_score", "count"),
          average_uplift=("uplift_score", "mean"),
          actual_conversion=("conversion", "mean")
      )
      .reindex(segment_order)
)

segment_summary["average_uplift"] *= 100
segment_summary["actual_conversion"] *= 100

col1, col2 = st.columns(2)

with col1:
    st.dataframe(
        segment_summary.round(3),
        use_container_width=True
    )

with col2:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        segment_summary.index,
        segment_summary["average_uplift"]
    )

    ax.axhline(0, linewidth=1)

    ax.set_title("Average Predicted Uplift")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Predicted Uplift (%)")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig)

# -----------------------------
# Customer Explorer
# -----------------------------
st.header("🔎 Customer Uplift Explorer")

selected_segment = st.selectbox(
    "Select Uplift Segment",
    ["All"] + segment_order
)

if selected_segment == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["uplift_segment"] == selected_segment
    ]

st.write(
    f"Showing **{len(filtered_df):,} customers**"
)

display_columns = [
    "segment",
    "treatment",
    "conversion",
    "treatment_probability",
    "control_probability",
    "uplift_score",
    "uplift_segment"
]

st.dataframe(
    filtered_df[display_columns]
    .sort_values("uplift_score", ascending=False)
    .head(100)
    .round(4),
    use_container_width=True
)

# -----------------------------
# Business Interpretation
# -----------------------------
st.header("💡 Business Interpretation")

st.markdown(
    """
    **High Uplift:** Customers with the highest predicted incremental response.

    **Moderate Uplift:** Customers with a moderate predicted treatment effect.

    **Low Uplift:** Customers with relatively small predicted incremental response.

    **Negative Uplift:** Customers for whom the current model predicts a
    negative treatment effect.

    These are model-based predictions and should be validated with further
    experimentation before being used for production campaign decisions.
    """
)

# -----------------------------
# Model Evaluation
# -----------------------------
st.header("📈 Model Evaluation")

evaluation_data = pd.DataFrame({
    "Model": [
        "Random Forest Two-Model",
        "Logistic Regression Two-Model",
        "Class Transformation"
    ],
    "Qini AUC": [
        -0.122374,
        -0.000767,
        -0.098238
    ],
    "AUUC": [
        -0.005861,
        -0.000080,
        -0.004700
    ]
})

st.dataframe(
    evaluation_data,
    use_container_width=True
)

st.info(
    "The current models are experimental baselines. "
    "The slightly negative Qini/AUUC values indicate that further "
    "uplift-model optimization and validation are needed."
)

st.caption(
    "Experimentation & Uplift Modeling for Targeted Marketing"
)
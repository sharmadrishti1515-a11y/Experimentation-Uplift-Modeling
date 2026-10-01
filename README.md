# Experimentation & Uplift Modeling for Targeted Marketing

## 📌 Project Overview

This project uses **A/B testing and uplift modeling** to analyze the incremental impact of email marketing campaigns on customer conversion.

Traditional machine learning predicts **which customers are likely to purchase**. Uplift modeling goes one step further by estimating **which customers are more likely to purchase because of a marketing treatment**.

The project analyzes an email marketing experiment containing three groups:

* **No E-Mail** — Control group
* **Mens E-Mail** — Treatment group 1
* **Womens E-Mail** — Treatment group 2

The goal is to understand campaign effectiveness and identify customer segments with different predicted responses to marketing treatment.

---

## 🎯 Objectives

* Measure the effect of email campaigns on customer conversion.
* Compare treatment groups with the control group.
* Perform statistical analysis of experimental results.
* Build customer-level uplift models.
* Compare different uplift modeling approaches.
* Evaluate models using **Qini AUC** and **AUUC**.
* Segment customers according to predicted uplift.
* Generate business-oriented insights for targeted campaign optimization.

---

## 📊 Dataset

The project uses the **Hillstrom/MineThatData email marketing dataset**.

Dataset characteristics:

* **64,000 customer records**
* **12 original features**
* Marketing treatment information
* Customer purchase/conversion outcomes
* Historical customer behavior
* Demographic and channel information

### Main Features

| Feature           | Description                           |
| ----------------- | ------------------------------------- |
| `recency`         | Number of months since last purchase  |
| `history_segment` | Historical spending category          |
| `history`         | Historical customer spending          |
| `mens`            | Previous men's merchandise activity   |
| `womens`          | Previous women's merchandise activity |
| `zip_code`        | Customer geographic category          |
| `newbie`          | Whether the customer is new           |
| `channel`         | Customer's preferred purchase channel |
| `segment`         | Experiment group                      |
| `visit`           | Website visit outcome                 |
| `conversion`      | Purchase/conversion outcome           |
| `spend`           | Customer spending                     |

---

## 🧪 Experiment Design

The original experiment contains three groups:

```text
                    Email Experiment
                         │
          ┌──────────────┼──────────────┐
          │              │              │
      No E-Mail      Mens E-Mail    Womens E-Mail
       Control       Treatment 1     Treatment 2
```

For the binary uplift modeling experiments, the two email groups were initially combined into an **any-email treatment** and compared with the No E-Mail control group.

The original three-group structure was retained for descriptive experiment analysis.

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

* Loaded the raw CSV dataset using pandas.
* Checked dataset dimensions and data types.
* Checked missing values.
* Checked duplicate rows without blindly removing them because the dataset does not contain an explicit customer ID.
* Examined categorical variables and their distributions.
* Excluded post-treatment variables such as `visit` from modeling features.
* Excluded outcome variables such as `conversion` and `spend` from predictors.
* Converted categorical variables using One-Hot Encoding.
* Split the data into training and testing sets using stratified sampling.

### Modeling Features

The uplift models used:

```text
recency
history_segment
history
mens
womens
zip_code
newbie
channel
```

---

## 📈 Experiment Results

The observed conversion rates were:

| Experiment Group | Customers | Conversions | Conversion Rate |
| ---------------- | --------: | ----------: | --------------: |
| No E-Mail        |    21,306 |         122 |         0.5726% |
| Mens E-Mail      |    21,307 |         267 |         1.2531% |
| Womens E-Mail    |    21,387 |         189 |         0.8837% |

Compared with the control group:

* **Mens E-Mail:** +0.6805 percentage points observed uplift
* **Womens E-Mail:** +0.3111 percentage points observed uplift

A statistical test comparing the men's email group with the control group produced a very small p-value, indicating a statistically detectable difference in conversion rates for those groups in this experiment.

These are **group-level experimental results** and should not be interpreted as individual customer causal effects.

---

## 🤖 Uplift Modeling

The project experimented with multiple uplift modeling approaches.

### 1. Two-Model Approach — Random Forest

Two separate models were trained:

```text
Treatment Model → P(Conversion | Treatment)
Control Model   → P(Conversion | Control)
```

Individual predicted uplift:

```text
Uplift = P(Conversion | Treatment)
       - P(Conversion | Control)
```

### 2. Two-Model Approach — Logistic Regression

A second Two-Model implementation used Logistic Regression to generate smoother treatment and control probability estimates.

### 3. Class Transformation

A propensity-adjusted transformed outcome was used with a Random Forest Regressor to estimate individual uplift.

---

## 📊 Model Evaluation

Uplift models were evaluated using:

* **Qini AUC**
* **AUUC (Area Under the Uplift Curve)**

| Model                         |  Qini AUC |       AUUC |
| ----------------------------- | --------: | ---------: |
| Random Forest Two-Model       |  -0.12237 |   -0.00586 |
| Logistic Regression Two-Model | -0.000767 | -0.0000799 |
| Class Transformation          |  -0.09824 |   -0.00470 |

The Logistic Regression Two-Model produced the strongest result among the tested approaches on the current test split, although its Qini AUC and AUUC remained slightly negative.

Therefore, the current model should be considered an **experimental baseline rather than a production-ready uplift model**.

---

## 👥 Customer Uplift Segmentation

The Logistic Regression uplift scores were used to create four customer segments.

| Segment         | Customers | Average Predicted Uplift | Actual Conversion |
| --------------- | --------: | -----------------------: | ----------------: |
| Negative Uplift |       780 |                  -0.191% |            0.897% |
| Low Uplift      |     7,875 |                  +0.327% |            0.851% |
| Moderate Uplift |     3,092 |                  +0.645% |            0.809% |
| High Uplift     |     1,053 |                  +1.485% |            1.614% |

The **High Uplift** segment contains 1,053 customers with the highest predicted incremental response according to the current model.

These segments represent **model predictions**, not confirmed individual-level causal effects.

---

## 💡 Business Interpretation

The analysis demonstrates why looking only at overall conversion probability can be insufficient for campaign targeting.

A customer may have a high probability of purchasing regardless of whether they receive an email. Uplift modeling attempts to identify customers whose behavior is more likely to change because of the treatment.

Potential business applications include:

* Prioritizing customers for marketing campaigns.
* Reducing unnecessary campaign exposure.
* Identifying customers with potentially negative treatment response.
* Improving marketing resource allocation.
* Supporting personalized campaign strategies.

Because the current uplift models show weak Qini/AUUC performance, these business applications should be treated as **prototype use cases** until the model is further validated and improved.

---

## 📊 Project Visualizations

The project generates the following visualizations:

### Experiment Conversion Rate

```text
results/figures/experiment_conversion_rate.png
```

### Qini Curve

```text
results/figures/qini_curve_logistic.png
```

### Uplift Segmentation

```text
results/figures/uplift_segments.png
```

---

## 📁 Project Structure

```text
Experimentation-Uplift-Modeling/
│
├── data/
│   ├── raw/
│   │   └── hillstrom.csv
│   └── processed/
│       └── customer_uplift_results.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ab_testing.ipynb
│   ├── 04_uplift_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
├── models/
│
├── results/
│   └── figures/
│       ├── experiment_conversion_rate.png
│       ├── qini_curve_logistic.png
│       └── uplift_segments.png
│
├── dashboard/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Scikit-uplift
* SciPy
* Statsmodels
* Matplotlib
* Seaborn
* Jupyter Notebook
* Git & GitHub

---

## 🚀 Future Improvements

Future versions of the project can improve the current uplift modeling framework by:

* Treating Men's and Women's Email as separate treatments throughout modeling.
* Testing additional uplift algorithms.
* Hyperparameter tuning.
* Probability calibration.
* Cross-validation for more reliable model evaluation.
* Testing multiple train/test splits.
* Adding treatment propensity diagnostics.
* Improving Qini/AUUC performance.
* Building an interactive dashboard.
* Adding campaign cost and ROI analysis.
* Deploying the final model as an API or application.

---

## 📌 Key Learning

This project demonstrates the difference between **predictive modeling and causal/uplift-oriented modeling**.

Instead of asking:

> "Who is likely to purchase?"

the project focuses on:

> "Who is more likely to respond differently because of the marketing treatment?"

This distinction can help organizations move from broad customer targeting toward more data-driven experimental campaign optimization.

---

## 👩‍💻 Project Status

**Status: In Progress**

Core experimentation, uplift modeling, model evaluation, customer segmentation, and result visualization have been completed.

The next phase focuses on building an interactive dashboard and improving the uplift modeling methodology.

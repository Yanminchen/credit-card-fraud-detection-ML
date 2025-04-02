
# 🕵️‍♀️ Credit Card Fraud Detection - ML Project

## 📌 Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques and provide actionable business insights. The goal is to maximize fraud detection accuracy while minimizing disruption to genuine transactions. The final model can catch **53% of frauds by flagging the top 3% of risky transactions**, with an estimated **annual savings of $19.33 million**.

---

## 🧠 Objectives

- Understand and preprocess transaction data
- Create meaningful time-based and behavior-based features
- Perform feature selection to avoid overfitting in high dimensions
- Compare multiple machine learning models
- Provide business-oriented evaluation of fraud detection performance

---

## 📂 Project Structure

```
credit_card_fraud_detection_ml/
├── data/
│   ├── cleaned_data_final.csv
│   ├── filter_top.csv
│   ├── vars_final.csv
│   ├── vars_keep_filter.csv
│   └── vars_keep_sorted.csv
│
├── images/
│   ├── feature_selection_graph_summary.png
│   └── summary.pdf
│
├── notebooks/
│   ├── applications_models.ipynb
│   ├── data_cleaning_and_exploration.ipynb
│   ├── feature_deduplication.ipynb
│   ├── feature_selection.ipynb
│   └── variable_creation.ipynb
│
├── requirements.txt
└── README.md
```

---

## 📊 Data Summary

- **Records**: 96,753
- **Period**: Jan 2010 – Dec 2010
- **Original Fields**: 10 (e.g., Date, Amount, Merchant Info, Fraud Label)
- **Engineered Features**: Over 1,400 (time-based aggregation, velocity/acceleration, variability, Benford’s law, etc.)
- **Target Variable**: `Fraud` (0 = not fraud, 1 = fraud)

---

## 🔧 Workflow

### 1. Data Cleaning
- Converted data types (e.g., Date to datetime)
- Removed extreme outlier transactions
- Filled nulls in merchant number/state/zip using mapping dictionaries
- Labeled foreign or unknown merchants as needed

### 2. Variable Creation
- Created 1,400+ features including:
  - Aggregated transaction statistics over multiple time windows
  - Behavioral features like velocity and acceleration
  - Zip-code and merchant-level trend variables
  - Variability over time

### 3. Feature Selection
- Applied forward/backward feature selection using:
  - LightGBM
  - Random Forest
- Final feature set: top 20 variables with highest KS scores

### 4. Model Exploration
Compared performance of:
- ✅ Logistic Regression (baseline)
- ✅ Random Forest (final choice)
- ✅ LightGBM (overfitting risk)
- ✅ Neural Network
- ✅ Decision Tree

**Final model**:  
```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=30,
    criterion='entropy'
)
```

---

## 📈 Results

| Dataset      | Score  |
|--------------|--------|
| Training     | 0.774  |
| Testing      | 0.761  |
| Out-of-Time  | 0.530  |

- 🚨 **53% of frauds caught** by flagging only 3% of transactions
- 💰 Estimated business savings: **$19.33 million/year**
- 🧾 Cost assumptions:
  - Gain: $400 per fraud caught
  - Cost: $20 per false positive

---

## 📌 Key Business Takeaways

- Fraud detection can be highly effective with engineered behavior-based features
- Business-aligned cost-benefit cutoff is critical for operational implementation
- Regular retraining and feature updates are essential due to evolving fraud patterns

---

## 👤 Author

**Yanmin Chen**  
🎓 Business Analytics @ USC Marshall  
📫 yanminc7@gmail.com 

---

## 🛠 Dependencies

To be listed in `requirements.txt`, but may include:

```txt
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
```

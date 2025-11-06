# Insurance Claims Analysis (PySpark Project)

This project explores and models an **insurance claims dataset** using **PySpark** for distributed data processing and analysis.  
The goal is to understand key patterns in vehicle insurance data and build a foundation for claim prediction.

---

## Project Structure

insurance-claims-analysis/
├── main.py
├── data/
│   ├── raw/
│   └── preprocessed/
├── notebooks/
│   └── 01_EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md


---

## Current Progress

### Step 1: Environment Setup
- Created virtual environment (`venv`)
- Installed dependencies (PySpark, Pandas, Matplotlib, Seaborn)
- Initialized Git repository and `.gitignore`

### Step 2: Data Loading & Initial EDA
- Loaded Kaggle dataset (`insurance_claims.csv`) using PySpark
- Verified schema and data types
- Confirmed **no missing values**
- Explored **target variable (`claim_status`)**
  - 0 → 93.6% (no claim)
  - 1 → 6.4% (claim)
  - → *Highly imbalanced dataset*
- Visualized **distributions of 12 numeric features**
- Documented feature types and potential preprocessing needs

---

## Key Insights So Far
- Data quality: No missing values detected.
- Target imbalance: Significant (only ~6% claims).
- Several features require cleaning (e.g., `max_torque`, `max_power`).
- Many “Yes/No” columns need binary encoding.
- Possible multicollinearity between size-related variables (`length`, `width`, `gross_weight`).

---

## Next Steps
**Step 3: Data Preprocessing**
- Convert column types (string → numeric)
- Encode binary categorical features (`is_*` columns)
- Parse mixed-format features (`max_torque`, `max_power`)
- Drop irrelevant IDs (`policy_id`)
- Save a clean dataset for feature engineering

---

## Tech Stack
- **Language:** Python 3.12  
- **Core Libraries:** PySpark, Pandas, Matplotlib, Seaborn  
- **Environment:** VS Code  
- **Dataset:** [Kaggle – Insurance Claims Dataset](https://www.kaggle.com/datasets/litvinenko630/insurance-claims)

---

## ✏️ Author
Project by *Maryna Kyslytsyna* — built for portfolio and PySpark practice.


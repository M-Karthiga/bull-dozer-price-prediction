#  Bulldozer Sale Price Prediction

An end-to-end machine learning regression project that predicts the **auction sale price of heavy construction equipment (bulldozers)**, built on the [Kaggle Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers) competition dataset.

Three models are trained and compared — Random Forest, XGBoost, and CatBoost — using the competition's official evaluation metric (RMSLE).

---

##  Problem Definition

> Given a bulldozer's characteristics and historical sale prices of similar equipment, how accurately can we predict its future auction sale price?

This is a **time series regression** problem. The model is trained on auction records through the end of 2011, and evaluated on data from January–April 2012 — mirroring Kaggle's own train/test split and simulating a real-world scenario where the model must generalise to future, unseen sales.

---

##  Dataset

- **Source:** [Kaggle — Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers)
- ~400,000 auction records with 50+ features covering equipment type, configuration, usage, and sale conditions
- Three splits: `Train.csv` (pre-2012), `Valid.csv` (Jan–Apr 2012), `Test.csv` (held-out, labels withheld by Kaggle)

> Raw data is not included in this repo due to Kaggle's terms and file size. See [Setup](#-setup--reproduction) to download it.

---

##  Approach

### 1. Exploratory Data Analysis
- Distribution of sale prices (right-skewed — log transform used for GBM training)
- Missing value analysis across 50+ columns
- Time-based patterns extracted from `saledate`

### 2. Feature Engineering
- Decomposed `saledate` into `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, `saleDayOfYear`
- Sorted data chronologically before splitting to respect time series structure

### 3. Preprocessing
- Converted all string columns to ordered pandas `category` dtype
- Filled missing numeric values with column median; added a binary `_is_missing` flag per affected column
- Encoded categorical columns as integer codes (+1 to avoid -1 for missing categories)

### 4. Modelling
Three models trained and compared:

**Random Forest** — `RandomForestRegressor`
- Used `max_samples=10000` for fast hyperparameter search with `RandomizedSearchCV` (20 iterations, 5-fold CV)
- Final model trained on full dataset with best-found parameters (`n_estimators=40`, `max_features=0.5`, `min_samples_split=14`)

**XGBoost** — `XGBRegressor`
- Trained on `log1p`-transformed target (prices are right-skewed; log transform stabilises predictions and prevents negative outputs)
- Used early stopping (`early_stopping_rounds=50`) to automatically find optimal `n_estimators` — more efficient than grid search for sequential boosting models
- Converged at 1997 trees

**CatBoost** — `CatBoostRegressor`
- Also trained on `log1p`-transformed target with early stopping
- Converged at 4063 trees

### 5. Evaluation
- **Primary metric:** Root Mean Squared Log Error (RMSLE) — the competition's official metric
- Also tracked: MAE, R², and overfitting gap (Val RMSLE - Train RMSLE)

### 6. Feature Importance
- Identified strongest predictors of sale price using `feature_importances_`
- Top features: `YearMade`, `ProductSize`, `Enclosure`, `saleYear`

---

## 📈 Results

| Metric | Random Forest | XGBoost | CatBoost |
|---|---:|---:|---:|
| Training MAE | $2,954 | $4,042 | $4,359 |
| **Validation MAE** | $5,951 | **$5,907** | $5,967 |
| Training RMSLE | 0.145 | 0.189 | 0.204 |
| **Validation RMSLE** | 0.245 | **0.235** | 0.237 |
| Training R² | 0.959 | 0.923 | 0.909 |
| **Validation R²** | **0.882** | 0.876 | 0.873 |
| Overfit Gap (Val - Train RMSLE) | 0.100 | **0.046** | 0.033 |

**XGBoost achieves the best validation RMSLE (0.235)**, the competition's official metric.

Random Forest has the highest R² (0.882) but also the largest overfitting gap (0.100) — it memorises training data more than the boosting models.

All three models are within 0.01 RMSLE of each other, which suggests the **preprocessing and feature engineering are the primary performance drivers**, not the choice of model.

---

##  Tech Stack

| Library | Purpose |
|---|---|
| pandas / NumPy | Data loading, feature engineering, missing value handling |
| scikit-learn | Random Forest, RandomizedSearchCV, evaluation metrics |
| XGBoost | Gradient boosted trees with early stopping |
| CatBoost | Gradient boosted trees with native categorical support |
| Matplotlib | Visualisation, feature importance plot |

---

## 📁 Project Structure

```
.
├── bulldozer-price-regression.ipynb   # Main notebook: EDA → preprocessing → modelling → evaluation
├── data/                              # (not tracked) raw CSVs from Kaggle
├── requirements.txt                   # Python dependencies
└── README.md
```

---

##  Setup & Reproduction

```bash
git clone https://github.com/<your-username>/bulldozer-price-prediction.git
cd bulldozer-price-prediction

python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Download `Train.zip`, `Valid.zip`, and `Test.zip` from the [Kaggle competition page](https://www.kaggle.com/c/bluebook-for-bulldozers/data) (free Kaggle account required). Unzip into a `data/` folder, then launch the notebook:

```bash
jupyter notebook bulldozer-price-regression.ipynb
```

---


---

##  Acknowledgements

Dataset provided by Fast Iron and Kaggle as part of the [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers) competition.

---

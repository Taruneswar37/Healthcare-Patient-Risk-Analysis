# 🏥 Healthcare Patient Risk Analysis
### AI/ML Assessment — Novintix

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project performs a comprehensive AI/ML analysis on a real-world style healthcare patient dataset containing **55,500 records**. The pipeline covers four stages — exploratory analysis, supervised classification, unsupervised anomaly detection, and an AI-powered doctor recommendation system using Google Gemini.

---

## 📂 Project Structure
```
Healthcare-Patient-Risk-Analysis/
│
├── data/
│   └── healthcare_dataset.csv         ← Download from Kaggle (link below)
│
├── outputs/                           ← Auto-generated plots saved here
│
├── task1_eda.py                       ← Exploratory Data Analysis
├── task2_supervised_learning.py       ← Test Result Classification
├── task3_anomaly_detection.py         ← Billing Anomaly Detection
├── task4_ai_doctor_recommendation.py  ← Gemini AI Recommendations
│
├── .env                               ← Your Gemini API key (never pushed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle — Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- **Records:** 55,500 patients
- **Features:** 15 columns including Age, Gender, Blood Type, Medical Condition, Billing Amount, Admission Type, Medication, Test Results, and more
- **No missing values** across any column

---

## 🔬 Approach & Tasks

---

### ✅ Task 1 — Exploratory Data Analysis (`task1_eda.py`)

**Approach:**
- Loaded and inspected the dataset for shape, data types, and missing values
- Analyzed numerical distributions using histograms with KDE curves and box plots
- Visualized categorical frequencies using bar charts and pie charts
- Performed cross-feature analysis to find relationships between variables

**Key Findings:**
- Age is uniformly distributed between 13–89 years (mean: 51.5)
- Billing Amount ranges from **-$2,008 to $52,764** — the negative value is a data anomaly investigated in Task 3
- Medical Conditions, Admission Types, and Medications are all fairly balanced across categories
- No missing values detected across all 55,500 records

**Visualizations Generated:**
- Distribution histograms for Age, Billing Amount, Room Number
- Box plots for outlier detection
- Bar charts and pie charts for categorical features
- Age by Medical Condition (box plot)
- Billing Amount by Admission Type (violin plot)
  
**Note:** Once you see the one visualization close it to move to next.

---

### ✅ Task 2 — Supervised Learning (`task2_supervised_learning.py`)

**Objective:** Predict `Test Results` — a 3-class target: `Normal`, `Abnormal`, `Inconclusive`

**Approach:**
1. **Feature Engineering** — Extracted `Length of Stay`, `Admission Month`, and `Admission DayOfWeek` from date columns
2. **Encoding** — Applied `LabelEncoder` to all categorical features
3. **Train/Test Split** — 80/20 stratified split (44,400 train / 11,100 test)
4. **Model Training** — Trained and compared 3 models:
   - Random Forest *(primary)*
   - Logistic Regression
   - Gradient Boosting
5. **Evaluation** — Accuracy, Classification Report, Confusion Matrix, Feature Importance, 5-Fold Cross Validation

**Results:**

| Model | Accuracy |
|---|---|
| **Random Forest** | **38.63%** ✅ Best |
| Logistic Regression | 33.4% |
| Gradient Boosting | 33.3% |

> **Note on Accuracy:** The dataset has 3 perfectly balanced classes, making the random baseline exactly **33.3%**. Random Forest scores **38.63%**, meaningfully above baseline. The dataset is synthetically generated with weak correlation between features and test results, which limits accuracy — confirmed by 5-Fold CV score of **39.99% ± 4.85%**.

**Visualizations Generated:**
- Model accuracy comparison bar chart
- Confusion matrix heatmap
- Feature importance chart
- Actual vs Predicted distribution comparison

**Note:** Once you see the one visualization close it to move to next.

---

### ✅ Task 3 — Unsupervised Learning / Anomaly Detection (`task3_anomaly_detection.py`)

**Objective:** Detect unusually high or low Billing Amount values

**Approach:**
Three methods were applied and compared:

| Method | Anomalies Found | Notes |
|---|---|---|
| Z-Score (\|z\| > 3) | 0 | Billing spread too uniform for threshold-based detection |
| IQR (1.5×IQR) | 0 | Same reason — wide uniform distribution |
| **Isolation Forest** | **2,775** ✅ | Detects multivariate anomalies effectively |

**Why Isolation Forest?**
Unlike Z-Score and IQR which operate on a single dimension, Isolation Forest analyzed **Billing Amount + Age + Room Number together**, catching anomalies that single-variable methods miss entirely.

**Anomaly Patterns Found:**
- 🔴 **Very young patients (13–22) with extremely high bills ($47K–$51K)** — clinically unusual
- 🔴 **Elderly patients (84–85) billed under $500** — suspiciously low for any admission
- 🔴 **Negative billing amounts (-$2,008)** — impossible in real healthcare, clear data errors

**Visualizations Generated:**
- Anomaly score distribution
- Billing vs Age scatter plot (normal vs anomalous highlighted)
- Normal vs Anomalous billing distribution overlay
- Method comparison bar chart
- Anomalous records by Medical Condition

**Note:** Once you see the one visualization close it to move to next.

---

### ✅ Task 4 — AI Task / LLM (`task4_ai_doctor_recommendation.py`)

**Objective:** Generate a doctor-style clinical recommendation using an LLM

**Approach:**
1. Re-trained the Random Forest model from Task 2 on the full dataset
2. For a given patient, predicted their `Test Result` with confidence score
3. Passed the predicted result + patient attributes to **Google Gemini 2.5 Flash**
4. Gemini generates a structured clinical recommendation

**Prompt Design:**
The prompt instructs Gemini to act as a clinical doctor and produce 5 structured sections:
- **Summary** — Plain-language interpretation of the predicted result
- **Clinical Recommendation** — Advised tests, medications, and lifestyle changes
- **Health Advice** — 3 specific tips relevant to the condition and age group
- **Follow-up** — Recommended review schedule
- **Risk Level** — Low / Moderate / High with explanation

**Sample Output:**
```
Patient         : Rajesh Kumar
Predicted Result: Abnormal
Confidence      : 33.7%

Dear Mr. Kumar,
[Gemini generates a full structured clinical recommendation...]
```

**API:** Google Gemini 2.5 Flash (free tier) via `google-genai` SDK

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Taruneswar37/Healthcare-Patient-Risk-Analysis.git
cd Healthcare-Patient-Risk-Analysis
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
- Go to [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- Download `healthcare_dataset.csv`
- Place it inside the `data/` folder

### 5. Configure Gemini API Key (Task 4 only)
- Get a **free** API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_actual_key_here
```

---

## ▶️ How to Run

Run each task independently from the terminal:
```bash
# Task 1 — EDA
python task1_eda.py

# Task 2 — Supervised Learning
python task2_supervised_learning.py

# Task 3 — Anomaly Detection
python task3_anomaly_detection.py

# Task 4 — AI Doctor Recommendation
python task4_ai_doctor_recommendation.py
```

All plots display as pop-up windows. **Close each plot window** to continue to the next one.

---

## 📦 Requirements
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
google-genai
python-dotenv
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy google-genai python-dotenv
```

---

## 📈 Results Summary

| Task | Approach | Key Result |
|---|---|---|
| EDA | Histograms, box plots, bar charts, violin plots | No missing values; negative billing detected |
| Supervised Learning | Random Forest (best of 3 models) | 38.63% accuracy — 5.3% above random baseline |
| Anomaly Detection | Isolation Forest (best of 3 methods) | 2,775 anomalies flagged (~5% of records) |
| AI Recommendation | Google Gemini 2.5 Flash | Structured clinical recommendations generated |

---


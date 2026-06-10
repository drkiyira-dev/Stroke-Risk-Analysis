# Stroke Risk Data Exploration & Analysis

> A modular Pandas-based data analysis project exploring stroke risk factors across 5,110 patient records.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Type](https://img.shields.io/badge/Type-University%20Assignment-orange)

---

## Overview

This project performs an end-to-end exploratory data analysis (EDA) on the
[Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
Starting from raw data inspection, through cleaning and preprocessing, to hypothesis testing
and a generated PDF report — all implemented in modular Python using Pandas.

**Key question:** Which factors are most associated with stroke risk?

---

## Dataset

- **Source:** [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Size:** 5,110 rows × 12 columns
- **Target column:** `stroke` (0 = No, 1 = Yes)
- **Overall stroke rate:** 4.87% (249 out of 5,110 patients)

> ⚠️ The raw CSV files are **not included** in this repository due to Kaggle's data license.
> Please download `healthcare-dataset-stroke-data.csv` from the link above and rename it
> to `中风预测.csv` before running.

---

## Project Structure

```
stroke-risk-analysis/
│
├── task1.py               # Task 1 — Data loading & inspection
│                          #   Outputs: shape, dtypes, missing values, head(5)
│
├── task2.py               # Task 2 — Preprocessing
│                          #   Drops 'id', fills BMI median, translates all
│                          #   column names and category values to Chinese
│                          #   Saves: 中风预测_预处理后.csv
│
├── task3.py               # Task 3 — Exploratory analysis & hypothesis testing
│                          #   Tests 3 hypotheses: Age / Heart Disease / Smoking
│                          #   Covers: .loc[], .groupby(), .describe(),
│                          #           .mean(), .count(), pd.cut(), apply()
│
├── main.py                # Main entry point — orchestrates Tasks 1–4
│
├── generate_report.py     # Standalone PDF report generator (fpdf2)
│                          #   Produces: 中风风险分析报告.pdf
│
└── 中风风险分析报告.pdf    # Generated analysis report (Chinese)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas fpdf2
```

### 2. Place the dataset

Download the CSV from Kaggle, rename it, and place it in the project root:

```
中风预测.csv   ← rename from healthcare-dataset-stroke-data.csv
```

### 3. Run the full analysis pipeline

```bash
python main.py
```

This will execute Tasks 1–4 in sequence and save `中风预测_预处理后.csv`.

### 4. Generate the PDF report (optional)

```bash
python generate_report.py
```

> **Note:** PDF generation uses the system font `STHeiti` (macOS).
> On Windows or Linux, update the `FONT_TTF` path in `generate_report.py`
> to a local CJK-compatible `.ttf` or `.ttc` font (e.g. WenQuanYi).

---

## Key Findings

Three hypotheses were tested. All results are drawn from real data.

### Hypothesis 1 — Older age is strongly associated with higher stroke rate ✅

| Age Group     | Samples | Stroke Cases | Stroke Rate |
|---------------|---------|--------------|-------------|
| Minor (< 18)  | 856     | 2            | 0.23%       |
| Young (18–40) | 1,314   | 6            | 0.46%       |
| Middle (40–60)| 1,564   | 60           | 3.84%       |
| Senior (≥ 60) | 1,376   | 181          | **13.15%**  |

Stroke rate increases monotonically with age. The senior group is **57× higher** than minors.

---

### Hypothesis 2 — Heart disease significantly raises stroke risk ✅

| Group              | Samples | Stroke Cases | Stroke Rate  |
|--------------------|---------|--------------|--------------|
| No heart disease   | 4,834   | 202          | 4.18%        |
| Has heart disease  | 276     | 47           | **17.03%**   |

Patients with heart disease have a stroke rate **4.1× higher** than those without.

---

### Hypothesis 3 — Smoking and stroke rate (counter-intuitive result) ⚠️

1,544 records with `Unknown` smoking status were filtered before analysis.

| Group           | Samples | Stroke Cases | Stroke Rate |
|-----------------|---------|--------------|-------------|
| Never smoked    | 1,892   | 90           | 4.76%       |
| Currently smokes| 789     | 42           | 5.32%       |
| Formerly smoked | 885     | 70           | **7.91%**   |

**Unexpected:** Former smokers show a *higher* rate than current smokers.
This is likely a **confounding variable** effect — former smokers tend to be older,
and age is a far stronger risk factor. This result highlights that
**correlation ≠ causation** in single-variable analysis.

---

## Tech Stack

| Tool       | Usage                                      |
|------------|--------------------------------------------|
| Python 3   | Core language                              |
| Pandas     | Data loading, cleaning, groupby, describe  |
| fpdf2      | PDF report generation                      |
| STHeiti / WenQuanYi | CJK font rendering in PDF        |

---

## Preprocessing Summary

| Step | Action |
|------|--------|
| Drop column | Removed `id` (no analytical value) |
| Fill missing | BMI: 201 missing values filled with median (28.10) |
| Rename columns | All 11 columns translated to Chinese |
| Remap values | 5 categorical columns translated (gender, job, marriage, residence, smoking) |

---

## Notes

- This project was completed as a university group assignment for a Pandas data analysis course (2026).
- The modular design (one file per task) maps directly to the four assignment tasks.
- The `是否中风` (stroke) column is kept as `0/1` integers in memory for numerical operations,
  and only converted to `是/否` (Yes/No) in the exported CSV for readability.

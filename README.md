# 📊 Election Analytics and Booth-Level Vote Share Modelling

## Project Overview

This project applies machine learning and deep learning to booth-level election data for **historical model evaluation and descriptive analytics**.

Historical 2016 election information is used to model 2021 party vote share. The project also includes descriptive analysis of the supplied 2026 booth dataset.

> **Important:** This repository should not be interpreted as a guaranteed prediction of any future election result.

## Models Used

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- MLP Deep Learning

## Historical Backtest Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.030072 | 0.054843 | 0.921065 |
| XGBoost | 0.031425 | 0.057781 | 0.912380 |
| Gradient Boosting | 0.032926 | 0.059705 | 0.906448 |
| MLP Deep Learning | 0.032251 | 0.061637 | 0.900297 |
| Linear Regression | 0.034491 | 0.063389 | 0.894547 |

Random Forest produced the lowest RMSE in the historical 2016-to-2021 backtest.

## Project Structure

```text
Election-Analytics-Booth-Level-Modelling/
│
├── Election_Project.ipynb
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
├── database/
├── models/
├── reports/
└── results/
```

## Main Files

### `Election_Project.ipynb`
Contains the end-to-end analysis, historical modelling, model comparison, deep-learning workflow, and descriptive booth-level analysis.

### `app.py`
Streamlit dashboard with sections for:
- Project Overview
- Model Comparison
- Historical Predictions
- Deep Learning
- 2026 Booth Analysis
- Project Files

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Streamlit
- SQLite
- Joblib
- Jupyter Notebook

## Run the Notebook

```bash
pip install -r requirements.txt
jupyter notebook Election_Project.ipynb
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## Notes on Interpretation

The model metrics above come from the historical 2016-to-2021 backtest contained in this project.

The 2026 portion of the app is descriptive analysis of the supplied booth dataset and should not be treated as a forecast, endorsement, or guaranteed outcome.

## Project Status

- ✅ Historical model training
- ✅ Model comparison
- ✅ Deep-learning workflow
- ✅ Saved ML models
- ✅ Streamlit application
- ✅ Results and reports
- ✅ SQLite database
- ✅ 2026 booth descriptive analysis

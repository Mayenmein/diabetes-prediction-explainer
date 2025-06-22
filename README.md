# Diabetes Risk Prediction with Explainability and Counterfactual Insights

## Overview

This project uses the **Pima Indians Diabetes dataset** to build a machine learning model that predicts the likelihood of diabetes. Unlike traditional classifiers, this project goes beyond simple prediction by providing:

- **Explainable AI (XAI)** insights using SHAP/LIME to interpret model outputs.
- **Counterfactual Explanations** that answer “what needs to change to reduce diabetes risk?”

The goal is to simulate a **decision-support tool** that not only informs, but empowers individuals with actionable, personalized insights.

---

## Objectives

- Build a predictive model for diabetes risk.
- Use **SHAP** or **LIME** to explain individual predictions.
- Generate **counterfactuals** (e.g., “If BMI were lower, prediction = non-diabetic”).
- Package it into a user-friendly analysis notebook or app.

---

## Dataset

- **Source**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records**: ~768 samples
- **Features**: Age, BMI, Glucose, Blood Pressure, etc.
- **Target**: `Outcome` (1 = Diabetes, 0 = No Diabetes)

---

## Tech Stack

- Python 3.8+
- Pandas, NumPy, scikit-learn
- LIME (for XAI)
- DiCE / Alibi / Custom (for counterfactuals)
- Optional: Streamlit / Gradio for a web interface

---

## Model Pipeline

1. **Data Preprocessing**
2. **Exploratory Data Analysis**
3. **Model Training** (Logistic Regression, Random Forest, etc.)
4. **Explainability Analysis** (SHAP values)
5. **Counterfactual Generation**
6. **Evaluation & Reporting**

---

## Example Insight

> A patient with high glucose and BMI is predicted to have diabetes.  
> SHAP shows **glucose** is the main driver.  
> Counterfactual: “If BMI was 5 points lower and glucose 15 units lower, model predicts no diabetes.”

---

## Project Structure


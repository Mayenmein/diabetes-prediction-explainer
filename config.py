# config.py

import os

# === Paths === #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "diabetes.csv")  # You can rename if your dataset is different
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_diabetes.csv")

NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")  # Path to save the trained model
# === Model Params === #
RANDOM_STATE = 42
TEST_SIZE = 0.2

# For reproducibility
SEED = 42

TARGET = "Outcome"  # The target variable in the dataset
FEATURES = ['glucose_insulin_interaction',
 'Insulin',
 'insulin_sensitivity_index',
 'insulin_resistance_index',
 'SkinThickness',
 'bmi_glucose_interaction',
 'Glucose',
 'glucose_to_insulin_ratio',
 'glycemic_load',
 'glucose_2',
 'bmi_age_interaction',
 'DiabetesPedigreeFunction',
 'blood_pressure_age_interaction',
 'BMI',
 'BMI_2']

# SHAP config
SHAP_SAMPLE_SIZE = 100  # Limit sample size for faster SHAP explanation

# === Counterfactual Config === #
COUNTERFACTUAL_CONFIG = {
    "features_to_vary": ["Glucose", "BMI", "Insulin", "Age"],
    "max_changes": 2,
    "desired_class": 0,  # Non-diabetic outcome
}

# === Logging === #
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

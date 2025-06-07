import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import config
def load_data(file_path = config.RAW_DATA_PATH):
    """
    Load data from a specified file path.
    
    Args:
        file_path (str): The path to the data file.
        
    Returns:
        data (list): A list of data entries.
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=config.TARGET)  # Remove rows with missing values 
    return df

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix of the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to plot.
    """
    plt.figure(dpi = 120,figsize= (5,4))
    mask = np.triu(np.ones_like(df.corr(),dtype = bool))
    sns.heatmap(df.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_boxplot(df,column):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x =df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

def plot_histograms(df):
    df.hist(bins=30, figsize=(15, 10), layout=(3, 3))
    plt.tight_layout()
    plt.show()
 
def imputer(df,numerical_cols):
    """
    Imputes missing values in all numeric columns of a DataFrame 
    by the mean of each class defined by `target_col`.
    """
    df_imputed = df.copy() 
    
    for col in numerical_cols:
        if df[col].isnull().any():
            df_imputed[col] = df.groupby(config.TARGET)[col].transform(
                lambda x: x.fillna(x.mean())
            )
    return df_imputed

def oversample_data(X,y):
    """
    Oversample the minority class in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        target_column (str): The name of the target column.
        
    Returns:
        pd.DataFrame: The DataFrame with oversampled data.
    """
    
    from imblearn.over_sampling import ADASYN
    
    adasyn = ADASYN(random_state=config.RANDOM_STATE, sampling_strategy='minority')
    X_resampled, y_resampled = adasyn.fit_resample(X , y  )
   
    return X_resampled, y_resampled

def feature_engineering(df):
    """
    Perform feature engineering on the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        
    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    
    # Derived Features
    df['insulin_resistance_index'] = (df['Glucose'] * df['Insulin']) / 405
    df['glucose_to_insulin_ratio'] = df['Glucose'] / df['Insulin']
    df['obesity_indicator'] = np.where(df['BMI'] >= 30, 1, 0)

    # Interaction Features
    df['glucose_insulin_interaction'] = df['Glucose'] * df['Insulin']
    df['bmi_age_interaction'] = df['BMI'] * df['Age']
    df['Pregnancies_age_interaction'] = df['Pregnancies'] * df['Age']
    df['bmi_glucose_interaction'] = df['BMI'] * df['Glucose']
    df['blood_pressure_age_interaction'] = df['BloodPressure'] * df['Age']
    df['insulin_sensitivity_index'] = 1 / (df['Glucose'] * df['Insulin'])

    df['glucose_2'] = df['Glucose'] **2
    df['age_2'] = df['Age'] **2
    df['BMI_2'] = df['BMI'] **2

    # Metabolic Features
    df['metabolic_syndrome_score'] = np.where(df['BMI'] >= 30, 1, 0) + \
                                    np.where(df['BloodPressure'] >= 80, 1, 0) + \
                                    np.where(df['Glucose'] >= 140, 1, 0)

    df['diabetes_risk_factors_count'] = np.where(df['Age'] >= 45, 1, 0) + \
                                        np.where(df['BMI'] >= 30, 1, 0) + \
                                        np.where(df['Glucose'] >= 140, 1, 0)

    # Other Features
    df['glycemic_load'] = df['Glucose'] * 0.1  # Assuming a simple calculation for glycemic load 

    return df
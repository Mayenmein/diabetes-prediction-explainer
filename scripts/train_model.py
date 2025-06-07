from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, classification_report, confusion_matrix
import pandas as pd
import config
import seaborn as sns
import matplotlib.pyplot as plt

def split_data(df, target=config.TARGET, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    Split the DataFrame into training and testing sets.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        target (str): The target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test: Split data.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(model,X_train, y_train):
    """
    Train a Random Forest Classifier on the training data.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        
    Returns:
        model: Trained Random Forest model.
    """ 
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    
    Args:
        model: The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target variable.
        
    Returns:
        dict: Evaluation metrics including accuracy, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }

def hyperparameter_tuning(X_train, y_train, model, param_distributions, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        model: The model to tune.
        param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
        cv (int): Number of cross-validation folds.
        
    Returns:
        best_model: The best model found by GridSearchCV.
    """
    from scipy.stats import randint
    from sklearn.model_selection import RandomizedSearchCV
    random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=100,              # Number of parameter settings sampled
    cv=5,                    # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'        # or 'accuracy', 'f1', etc.
)
    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.2f}")
    return random_search.best_estimator_

from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', verbose=2, n_jobs=-1):
    """
    Performs GridSearchCV for any scikit-learn model.

    Parameters:
        model: A scikit-learn compatible estimator (e.g., RandomForestClassifier()).
        param_grid (dict): Dictionary of hyperparameters to search over.
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series or np.array): Target values.
        cv (int): Number of cross-validation folds. Default is 5.
        scoring (str): Scoring metric. Default is 'accuracy'.
        verbose (int): Verbosity level. Default is 2.
        n_jobs (int): Number of parallel jobs. Default is -1 (all cores).

    Returns:
        best_model: Estimator with best found parameters.
        grid_search: Full GridSearchCV object (for inspection or plotting).
    """

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    print(f"Best {scoring.capitalize()} Score: {grid_search.best_score_:.4f}")

    return best_model 

def plot_feature_importances(model, feature_names):
    """
    Plot feature importances from the trained model.
    
    Args:
        model: The trained model.
        feature_names (list): List of feature names.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()
    return pd.DataFrame({
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices]
    }).sort_values(by="Importance", ascending=False)
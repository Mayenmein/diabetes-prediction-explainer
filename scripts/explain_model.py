
import eli5
from eli5.sklearn import PermutationImportance

def get_permutation_importance(model, X_test, y_test, random_state=42):
    """
    Fit PermutationImportance wrapper using ELI5.
    
    Parameters:
    - model: Trained classifier (any type).
    - X_test: Scaled test features (numpy or pandas).
    - y_test: True labels.
    - random_state: Random seed for reproducibility.
    
    Returns:
    - perm: Trained PermutationImportance object.
    """
    perm = PermutationImportance(model, random_state=random_state)
    perm.fit(X_test, y_test)
    return perm

def show_feature_importance(perm, feature_names):
    """
    Display feature importances using ELI5.

    Parameters:
    - perm: Trained PermutationImportance object.
    - feature_names: List of feature names (from original X).
    
    Returns:
    - HTML visual output (inline in notebook)
    """
    return eli5.show_weights(perm, feature_names=feature_names)

# lime_explainer_utils.py

import lime
import lime.lime_tabular
import numpy as np

def create_lime_explainer(X_train_scaled, feature_names, class_names=['Non-Diabetic', 'Diabetic']):
    """
    Create a LIME tabular explainer.

    Parameters:
    - X_train_scaled: Scaled training features (numpy array).
    - feature_names: List of original feature names (e.g., X.columns).
    - class_names: List of class labels for prediction classes.

    Returns:
    - explainer: LimeTabularExplainer object.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    return explainer

from IPython.display import display, HTML

def explain_instance(explainer, model, instance, num_features=6, display_in_notebook=True):
    """
    Explain a single instance with LIME and optionally display in notebook.

    Parameters:
    - explainer: LimeTabularExplainer object.
    - model: Trained classifier with `.predict_proba()`.
    - instance: 1D numpy array (a row from scaled test set).
    - num_features: How many top features to show.
    - display_in_notebook: If True, displays explanation inline (Jupyter only).

    Returns:
    - html_exp: HTML string of the LIME explanation.
    """
    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features
    )
    html_exp = exp.as_html(show_table=True)
    
    if display_in_notebook:
        display(HTML(html_exp))
    
    return html_exp


def explain_and_save_multiple_instances(
    explainer,
    model,
    X_test_scaled,
    instance_indices,
    feature_names,
    output_dir="lime_outputs",
    num_features=6
):
    """
    Explain and save LIME outputs for multiple test instances.

    Parameters:
    - explainer: LimeTabularExplainer object.
    - model: Trained classifier with `.predict_proba()`.
    - X_test_scaled: Scaled test set (numpy array).
    - instance_indices: List of indices to explain (e.g., [0, 1, 2]).
    - feature_names: List of original column names.
    - output_dir: Folder to save the HTML files.
    - num_features: Top features to include in the explanation.

    Returns:
    - None (saves files to disk).
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for idx in instance_indices:
        exp = explainer.explain_instance(
            X_test_scaled.iloc[idx],
            model.predict_proba,
            num_features=num_features
        )
        file_path = os.path.join(output_dir, f"lime_instance_{idx}.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())

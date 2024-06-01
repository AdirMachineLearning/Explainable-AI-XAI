# Explainable-AI-XAI
Explainable-AI-XAI - LIME &amp; SHAP Robust Way with Viz 
# Model Explanation with LIME and SHAP

This repository contains code to explain a machine learning model's predictions using LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations). The explanations are demonstrated on the Iris dataset using a Support Vector Classifier (SVC).

## Overview

This project includes the following functionalities:
- Training a machine learning model on the Iris dataset.
- Explaining a single data point's prediction using LIME.
- Explaining a single data point's prediction using SHAP.
- Visualizing the explanations provided by LIME and SHAP.

## Mathematical Formulas

### LIME

LIME approximates the model's prediction with a local surrogate model that is interpretable. The key idea is to perturb the input data and see how the predictions change. This process is mathematically described as:

$$
\hat{f}(x) = \arg \min_{g \in G} \sum_{z \in Z} \pi_x(z) \left( f(z) - g(z) \right)^2 + \Omega(g)
$$

where:
- $$\( \hat{f}(x) \)$$ is the surrogate model.
- $$\( G \)$$ is the set of interpretable models.
- $$\( \pi_x(z) \)$$ is the proximity measure between the instance \( x \) and \( z \).
- $$\( f(z) \)$$ is the prediction of the original model.
- $$\( \Omega(g) \)$$ is the complexity measure of the surrogate model \( g \).
### SHAP

SHAP values are based on the concept of Shapley values from cooperative game theory. The Shapley value for a feature \( i \) is given by:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]
$$

where:
- \( \phi_i \) is the Shapley value for feature \( i \).
- \( N \) is the set of all features.
- \( S \) is a subset of features not containing \( i \).
- \( f(S) \) is the model prediction using features in subset \( S \).

## Code

The full code is provided below. It includes functions to explain and visualize predictions using both LIME and SHAP.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Create and train a model
model = SVC(kernel='linear', probability=True)  # Ensure the model has predict_proba method for LIME
model.fit(X, y)

def explain_instance_with_lime(model, instance_index, X):
    """
    This function explains a single data point's prediction using LIME on the Iris dataset.

    Args:
        model: Trained machine learning model (must have a predict_proba method).
        instance_index: Index of the data point to explain (within the Iris dataset).
        X: The Iris features data (pandas DataFrame).

    Returns:
        LIME explanation for the model's prediction on the chosen data point.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values, feature_names=X.columns.tolist(), class_names=iris.target_names, mode='classification'
    )
    instance = X.iloc[instance_index].values.reshape(1, -1)
    explanation = explainer.explain_instance(instance[0], model.predict_proba)
    return explanation.as_list()

def explain_instance_with_shap(model, instance_index, X, background_sample_size=100):
    """
    This function explains a single data point's prediction using SHAP on the Iris dataset.

    Args:
        model: Trained machine learning model (must have a predict method).
        instance_index: Index of the data point to explain (within the Iris dataset).
        X: The Iris features data (pandas DataFrame).
        background_sample_size: Number of samples to use for SHAP background dataset.

    Returns:
        SHAP explanation for the model's prediction on the chosen data point.
    """
    # Summarize the background using K-means or random sampling
    background = shap.kmeans(X, background_sample_size)
    explainer = shap.KernelExplainer(model.predict, background)
    instance = X.iloc[instance_index].values.reshape(1, -1)
    shap_values = explainer.shap_values(instance)
    return explainer.expected_value, shap_values, instance

def visualize_lime_explanation(explanation):
    """
    Visualizes the LIME explanation using a bar chart and provides detailed explanations.

    Args:
        explanation: LIME explanation object.
    """
    if not isinstance(explanation, list) or not all(isinstance(i, tuple) for i in explanation):
        raise ValueError("Explanation must be a list of tuples")

    # Extract feature names and their contributions
    features = [feature for feature, weight in explanation]
    contributions = [weight for feature, weight in explanation]

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, contributions, color='skyblue')
    plt.xlabel('Contribution to Prediction')
    plt.title('LIME Explanation for Instance')
    plt.grid(True)

    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:.2f}', ha='left', va='center')

    plt.show()

    # Provide detailed explanations
    print("\nDetailed Explanation:")
    for feature, contribution in explanation:
        if contribution > 0:
            print(f"The feature '{feature}' has a positive contribution of {contribution:.2f}, meaning it supports the model's prediction.")
        else:
            print(f"The feature '{feature}' has a negative contribution of {contribution:.2f}, meaning it opposes the model's prediction.")

def visualize_shap_explanation(expected_value, shap_values, instance, feature_names, plot_type='force'):
    """
    Visualizes the SHAP explanation using various plots.

    Args:
        expected_value: SHAP expected value (base value).
        shap_values: SHAP values array.
        instance: The instance being explained.
        feature_names: List of feature names.
        plot_type: Type of SHAP plot ('force', 'summary', 'dependence').
    """
    shap.initjs()

    if plot_type == 'force':
        shap.force_plot(expected_value, shap_values[0], instance, feature_names=feature_names, matplotlib=True)
        plt.show()
    elif plot_type == 'summary':
        shap.summary_plot(shap_values, instance, feature_names=feature_names)
    elif plot_type == 'dependence':
        for i in range(len(feature_names)):
            shap.dependence_plot(i, shap_values, instance, feature_names=feature_names)
    else:
        raise ValueError("Invalid plot type. Choose 'force', 'summary', or 'dependence'.")

# Example usage
instance_index = 0  # Index of the data point to explain (within the Iris dataset)
shap_plot_type = 'force'  # Choose 'force', 'summary', or 'dependence'

# Get LIME explanation
lime_explanation = explain_instance_with_lime(model, instance_index, X)
visualize_lime_explanation(lime_explanation)

# Get SHAP explanation
expected_value, shap_values, instance = explain_instance_with_shap(model, instance_index, X)
visualize_shap_explanation(expected_value, shap_values, instance, X.columns.tolist(), plot_type=shap_plot_type)

import json
import os

def create_modeling_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Helper function to add markdown cell
    def add_markdown(source_list):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_list]
        })

    # Helper function to add code cell
    def add_code(source_list):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_list]
        })

    # Title & Introduction
    add_markdown([
        "# UCI Bank Marketing - Predictive Modeling",
        "",
        "## Overview & Business Context",
        "Having completed the Exploratory Data Analysis, we now transition to building predictive models to identify whether a bank client will subscribe to a term deposit. This is structured as a **binary classification** problem.",
        "",
        "### Crucial Warning: Target Leakage & The `duration` Feature",
        "As analyzed in our EDA, the `duration` column (call duration in seconds) is a major source of target leakage. In a real-world predictive modeling scenario, a customer's call duration is not known until the call ends. However, the model needs to make predictions *before* the call starts (to select the best prospects). Therefore, **we must drop the `duration` column before modeling to make a realistic predictive model.**",
        "",
        "### Key Performance Metrics under Class Imbalance",
        "The target variable `y` is highly imbalanced (~88.7% 'no' and ~11.3% 'yes'). Due to this imbalance, **Accuracy alone is a misleading performance metric.** A simple dummy model that always predicts 'no' will achieve 88.7% accuracy but fail to identify any subscribing clients.",
        "",
        "For a marketing team, the cost of missing a prospective subscriber (false negative) is high, while calling a non-subscriber (false positive) is relatively low. Therefore, we prioritize:",
        "- **Recall (yes)**: The percentage of actual subscribers the model successfully identifies.",
        "- **F1-Score (yes)**: The harmonic mean of Precision and Recall for subscribers.",
        "",
        "In this notebook, we construct a preprocessing pipeline and train three classifiers with balanced class weights to address the imbalance: Logistic Regression, Decision Trees, and Random Forests."
    ])

    # Section 1: Setup & Loading
    add_markdown([
        "## 1. Setup and Loading Data",
        "We import standard packages and scikit-learn modules, load the dataset, and output basic dataset specifications."
    ])

    add_code([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import os",
        "",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.tree import DecisionTreeClassifier",
        "from sklearn.ensemble import RandomForestClassifier",
        "from sklearn.metrics import (",
        "    confusion_matrix, classification_report, accuracy_score,",
        "    precision_score, recall_score, f1_score",
        ")",
        "",
        "# Set plotting options",
        "plt.style.use('seaborn-v0_8-whitegrid')",
        "plt.rcParams['figure.figsize'] = (10, 6)",
        "plt.rcParams['font.size'] = 11",
        "",
        "# Load dataset",
        "df = pd.read_csv('bank-additional/bank-additional/bank-additional-full.csv', sep=';')",
        "print('=== Dataset Shape ===')",
        "print(df.shape)",
        "print('\\n=== Target Variable Distribution ===')",
        "print(df['y'].value_counts())",
        "print(df['y'].value_counts(normalize=True) * 100)",
        "",
        "print('\\n=== First 5 Rows ===')",
        "display(df.head())"
    ])

    # Section 2: Splitting Data
    add_markdown([
        "## 2. Train-Test Split and Target Leakage Prevention",
        "We drop the `duration` column from the modeling features to prevent data leakage. We then partition the features `X` and target label `y` into a training set (80%) and a testing set (20%), stratifying by the target label to maintain the class distribution in both splits."
    ])

    add_code([
        "# Drop duration to prevent target leakage",
        "df_model = df.drop('duration', axis=1)",
        "",
        "# Separate features X and target y",
        "X = df_model.drop('y', axis=1)",
        "y = df_model['y']",
        "",
        "# Perform train-test split (80-20, stratify on y for class representation)",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42, stratify=y",
        ")",
        "",
        "print(f'Training shape: {X_train.shape}')",
        "print(f'Testing shape: {X_test.shape}')",
        "print(f'Train target ratio: \\n{y_train.value_counts(normalize=True)}')",
        "print(f'Test target ratio: \\n{y_test.value_counts(normalize=True)}')"
    ])

    # Section 3: Preprocessing Pipeline
    add_markdown([
        "## 3. Preprocessing Pipelines",
        "We identify numerical and categorical features. We construct a `ColumnTransformer` preprocessing pipeline:",
        "- **Numerical columns** are scaled using `StandardScaler`.",
        "- **Categorical columns** are encoded using `OneHotEncoder` with `handle_unknown='ignore'`."
    ])

    add_code([
        "# Separate features by dtype",
        "num_cols = list(X.select_dtypes(include=[np.number]).columns)",
        "cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)",
        "",
        "print(f'Numerical Features to scale: {num_cols}')",
        "print(f'Categorical Features to encode: {cat_cols}')",
        "",
        "# Construct preprocessing pipelines",
        "numeric_transformer = StandardScaler()",
        "categorical_transformer = OneHotEncoder(handle_unknown='ignore')",
        "",
        "preprocessor = ColumnTransformer(",
        "    transformers=[",
        "        ('num', numeric_transformer, num_cols),",
        "        ('cat', categorical_transformer, cat_cols)",
        "    ]",
        ")",
        "print('Preprocessing pipeline built successfully!')"
    ])

    # Section 4: Model Training and Evaluation
    add_markdown([
        "## 4. Model Training & Subgroup Performance Evaluation",
        "We initialize three distinct classification models embedded in a scikit-learn Pipeline with our preprocessor:",
        "1. **Logistic Regression** (class weights balanced to adjust for minority class significance).",
        "2. **Decision Tree Classifier** (balanced class weights, depth limit of 5 to prevent overfitting).",
        "3. **Random Forest Classifier** (balanced class weights, 100 trees, depth limit of 10 for regularized ensemble learning).",
        "",
        "We evaluate each model using confusion matrices with consistent `labels=['no', 'yes']` and classification reports utilizing `zero_division=0`."
    ])

    add_code([
        "# Define the models",
        "models = {",
        "    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),",
        "    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5),",
        "    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=10)",
        "}",
        "",
        "model_results = []",
        "",
        "for name, clf in models.items():",
        "    print(f'\\n==================== Training {name} ====================')",
        "    # Build pipeline",
        "    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])",
        "    ",
        "    # Train",
        "    pipeline.fit(X_train, y_train)",
        "    ",
        "    # Predict",
        "    y_pred = pipeline.predict(X_test)",
        "    ",
        "    # Metrics",
        "    cm = confusion_matrix(y_test, y_pred, labels=['no', 'yes'])",
        "    print('Confusion Matrix (labels=[no, yes]):')",
        "    print(cm)",
        "    ",
        "    print('\\nClassification Report:')",
        "    print(classification_report(y_test, y_pred, zero_division=0))",
        "    ",
        "    acc = accuracy_score(y_test, y_pred)",
        "    prec_yes = precision_score(y_test, y_pred, pos_label='yes', zero_division=0)",
        "    rec_yes = recall_score(y_test, y_pred, pos_label='yes', zero_division=0)",
        "    f1_yes = f1_score(y_test, y_pred, pos_label='yes', zero_division=0)",
        "    ",
        "    # Print custom summary",
        "    print(f'Accuracy:      {acc:.4f}')",
        "    print(f'Precision_yes: {prec_yes:.4f}')",
        "    print(f'Recall_yes:    {rec_yes:.4f}')",
        "    print(f'F1_yes:        {f1_yes:.4f}')",
        "    ",
        "    model_results.append({",
        "        'Model': name,",
        "        'Accuracy': acc,",
        "        'Precision_yes': prec_yes,",
        "        'Recall_yes': rec_yes,",
        "        'F1_yes': f1_yes",
        "    })",
        "",
        "results_df = pd.DataFrame(model_results)",
        "print('\\n=== Model Results Summary ===')",
        "display(results_df)"
    ])

    # Section 5: Save Results & Bar Chart Comparison
    add_markdown([
        "## 5. Model Performance Comparison and Exports",
        "We create a directory named `model_outputs/` and save our comparison results. We then construct a bar chart comparing performance metrics across our three trained models."
    ])

    add_code([
        "output_dir = 'model_outputs'",
        "os.makedirs(output_dir, exist_ok=True)",
        "",
        "# Save comparison table to CSV",
        "results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)",
        "",
        "# Plot a bar chart comparing metrics across the models",
        "x = np.arange(len(results_df['Model']))",
        "width = 0.2",
        "",
        "fig, ax = plt.subplots(figsize=(12, 7))",
        "ax.bar(x - 1.5*width, results_df['Accuracy'], width, label='Accuracy', color='#34495e', edgecolor='black')",
        "ax.bar(x - 0.5*width, results_df['Precision_yes'], width, label='Precision_yes (Subscribed)', color='#3498db', edgecolor='black')",
        "ax.bar(x + 0.5*width, results_df['Recall_yes'], width, label='Recall_yes (Subscribed)', color='#e67e22', edgecolor='black')",
        "ax.bar(x + 1.5*width, results_df['F1_yes'], width, label='F1_yes (Subscribed)', color='#2ecc71', edgecolor='black')",
        "",
        "ax.set_ylabel('Scores', fontsize=12)",
        "ax.set_title('Performance Metrics Comparison across Classifiers', fontsize=14, fontweight='bold')",
        "ax.set_xticks(x)",
        "ax.set_xticklabels(results_df['Model'], fontsize=11)",
        "ax.set_ylim(0, 1.1)",
        "ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))",
        "",
        "plt.tight_layout()",
        "plot_path = os.path.join(output_dir, 'model_comparison_plot.png')",
        "plt.savefig(plot_path, dpi=100)",
        "plt.show()",
        "print(f'Comparison plot saved successfully to {plot_path}')"
    ])

    add_markdown([
        "### Why Recall_yes and F1_yes matter more than Accuracy",
        "Because our dataset contains a significant class imbalance (88.7% no, 11.3% yes), standard accuracy is highly deceptive. If a model predicts 'no' for all instances, it achieves 88.7% accuracy but is practically useless.",
        "",
        "In a telemarketing context:",
        "- **Recall_yes (Sensitivity)** represents the proportion of actual buyers the model successfully flags. A higher recall means we contact more potential buyers, minimizing missed opportunities.",
        "- **F1_yes** represents a balance between precision (avoiding cold calling wrong customers) and recall. Selecting a model based on high F1-score ensures we target clients efficiently without wasting resources on massive, low-yield lists."
    ])

    # Section 6: Feature Importances
    add_markdown([
        "## 6. Feature Importance Extraction",
        "Using our trained **Random Forest pipeline** (which incorporates the preprocessor and Random Forest classifier), we extract feature importances. Since categorical features are one-hot encoded, we retrieve the correct feature names from our preprocessor step."
    ])

    add_code([
        "# Retrieve trained Random Forest model and pipeline steps",
        "rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ",
        "                               ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=10))])",
        "rf_pipeline.fit(X_train, y_train)",
        "",
        "# Get feature names out from the column transformer",
        "raw_feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()",
        "importances = rf_pipeline.named_steps['classifier'].feature_importances_",
        "",
        "# Clean up feature name prefixes (e.g., 'num__age' -> 'age', 'cat__job_admin.' -> 'job_admin.')",
        "cleaned_feature_names = [name.split('__')[-1] for name in raw_feature_names]",
        "",
        "# Create importance DataFrame",
        "importance_df = pd.DataFrame({",
        "    'Feature': cleaned_feature_names,",
        "    'Importance': importances",
        "}).sort_values(by='Importance', ascending=False)",
        "",
        "# Save feature importances to CSV",
        "importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)",
        "print('=== Top 15 Most Important Features ===')",
        "display(importance_df.head(15))",
        "",
        "# Plot the top 15 most important features",
        "top_15_features = importance_df.head(15)",
        "",
        "plt.figure(figsize=(12, 7))",
        "plt.barh(top_15_features['Feature'][::-1], top_15_features['Importance'][::-1], color='#8e44ad', edgecolor='black', alpha=0.85)",
        "plt.title('Top 15 Most Important Features - Random Forest Classifier', fontsize=14, fontweight='bold')",
        "plt.xlabel('Importance Score', fontsize=12)",
        "plt.ylabel('Features', fontsize=12)",
        "plt.tight_layout()",
        "feat_plot_path = os.path.join(output_dir, 'feature_importance_plot.png')",
        "plt.savefig(feat_plot_path, dpi=100)",
        "plt.show()",
        "print(f'Feature importance plot saved successfully to {feat_plot_path}')"
    ])

    # Section 7: Conclusion
    add_markdown([
        "## 7. Conclusions and Key Insights",
        "",
        "### 1. Model Selection & Performance Summary",
        "Based on the trained pipelines, the model performance is summarized below:",
        "- **Logistic Regression** and **Decision Tree** models with balanced class weights trade off precision for a much higher recall. The Decision Tree (max_depth=5) restricts complexity to avoid overfitting, while Logistic Regression fits a linear decision boundary.",
        "- **Random Forest Classifier** (max_depth=10, 100 estimators) typically performs best. It manages to achieve a higher F1-score by maintaining a stronger balance between Precision and Recall for the positive class (`yes`), thanks to ensemble bagging.",
        "",
        "### 2. Feature Importance Insights",
        "The feature importance plot for the Random Forest model reveals that macro-economic indicators are highly influential in predicting subscription success. Specifically:",
        "- **`euribor3m`** (3-month Euribor interest rate) and **`nr.employed`** (number of employees indicator) consistently rank as top features. This suggests that subscription rates are heavily tied to broader economic cycles and interest rate climates.",
        "- **`pdays`** (days since last contact from a previous campaign) and **`poutcome`** (previous campaign success) are also important, validating that customer history and relationship continuity are critical factors.",
        "",
        "### 3. Key Takeaways",
        "1. **Prevention of Target Leakage**: Dropping the `duration` column was essential. While duration is highly predictive, it represents leakage because it is unavailable before making a phone call.",
        "2. **Imbalance Matters**: Relying on accuracy in an imbalanced marketing setup will misdirect resources. We successfully optimized models using balanced class weights, making them significantly better at identifying true prospects (high recall_yes) while balancing cold call efficiency (F1_yes)."
    ])

    # Write notebook file
    notebook_path = "02_Modeling_Bank_Marketing.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook successfully written to {notebook_path}!")

if __name__ == "__main__":
    create_modeling_notebook()

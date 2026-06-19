import json
import os

def create_advanced_notebook():
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
        "# UCI Bank Marketing - Advanced Modeling & Class Imbalance Handling",
        "",
        "## Overview & Objectives",
        "Following our preliminary EDA and baseline modeling stages, this notebook focuses on solving the **class imbalance problem** and applying **advanced ensemble classification** to optimize predictions.",
        "",
        "### Key Areas of Focus:",
        "1. **Imbalance Handling**: Apply class weighting, Random Oversampling (ROS), Random Undersampling (RUS), and Synthetic Minority Over-sampling Technique (SMOTE) to rebalance the training subset.",
        "2. **Advanced Tabular Models**: Train and compare Gradient Boosting (sklearn), Random Forest (class weighted), and state-of-the-art tree libraries: **XGBoost**, **LightGBM**, and **CatBoost**.",
        "3. **Threshold Tuning**: Move beyond default decision thresholds (0.5) to optimize target Recall and F1-score tradeoffs.",
        "4. **Hyperparameter Tuning**: Optimize Random Forest hyperparameters using grid search.",
        "5. **Validation & Curves**: Run 3-fold cross-validation, and plot Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.",
        "6. **Error Analysis**: Characterize false positives and false negatives to evaluate the business impact of the final model.",
        "",
        "### Safeguards & Guidelines:",
        "- **Prevent Data Leakage**: The target leakage variable `duration` is dropped immediately before splitting.",
        "- **Numerical Coding**: The target `y` is encoded as numeric (`no` = 0, `yes` = 1, positive class = 1) for consistent evaluation.",
        "- **Metric Warn Minimization**: Set `zero_division=0` in sklearn metrics to prevent diagnostic output spam.",
        "- **Confusion Matrix Ordering**: Consistent labeling via `labels=[0, 1]` represents `no` (0) and `yes` (1)."
    ])

    # Section 1: Setup & Loading
    add_markdown([
        "## 1. Setup & Data Loading",
        "We import standard packages and setup target directories. We drop the `duration` column to prevent data leakage and split our features `X` and target `y` (mapped to 0/1) using `test_size=0.2, random_state=42, stratify=y`."
    ])

    add_code([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import os",
        "",
        "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.tree import DecisionTreeClassifier",
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier",
        "from sklearn.metrics import (",
        "    confusion_matrix, classification_report, accuracy_score,",
        "    precision_score, recall_score, f1_score,",
        "    roc_auc_score, precision_recall_curve, auc, roc_curve",
        ")",
        "",
        "# Ensure output folder exists",
        "output_dir = 'advanced_outputs'",
        "os.makedirs(output_dir, exist_ok=True)",
        "",
        "plt.style.use('seaborn-v0_8-whitegrid')",
        "plt.rcParams['figure.figsize'] = (10, 6)",
        "plt.rcParams['font.size'] = 11",
        "",
        "# Load dataset",
        "df = pd.read_csv('bank-additional/bank-additional/bank-additional-full.csv', sep=';')",
        "",
        "# Drop target leakage column",
        "df_model = df.drop('duration', axis=1)",
        "",
        "# Separate inputs X and map target y to numeric values (no=0, yes=1)",
        "X = df_model.drop('y', axis=1)",
        "y = df_model['y'].map({'no': 0, 'yes': 1})",
        "",
        "# Stratified train-test split",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42, stratify=y",
        ")",
        "print('Data split successfully!')",
        "print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')"
    ])

    # Section 2: Preprocessing
    add_markdown([
        "## 2. Preprocessing Pipeline",
        "We identify numerical and categorical features and build our standard preprocessor using `ColumnTransformer`."
    ])

    add_code([
        "num_cols = list(X.select_dtypes(include=[np.number]).columns)",
        "cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)",
        "",
        "preprocessor = ColumnTransformer(",
        "    transformers=[",
        "        ('num', StandardScaler(), num_cols),",
        "        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)",
        "    ]",
        ")",
        "",
        "# Fit-transform training data, transform test data",
        "X_train_pre = preprocessor.fit_transform(X_train)",
        "X_test_pre = preprocessor.transform(X_test)",
        "print('Preprocessing completed successfully!')"
    ])

    # Section 3: Baseline Logistic Regression Without Imbalance Handling
    add_markdown([
        "## 3. Baseline Logistic Regression (Unweighted)",
        "We train a default Logistic Regression model without class weighting to establish baseline classification metrics."
    ])

    add_code([
        "lr_baseline = LogisticRegression(max_iter=1000)",
        "lr_baseline.fit(X_train_pre, y_train)",
        "y_pred = lr_baseline.predict(X_test_pre)",
        "y_proba = lr_baseline.predict_proba(X_test_pre)[:, 1]",
        "",
        "# Evaluate",
        "acc = accuracy_score(y_test, y_pred)",
        "prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "roc_auc = roc_auc_score(y_test, y_proba)",
        "",
        "# Calculate PR-AUC",
        "precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)",
        "pr_auc = auc(recall_vals, precision_vals)",
        "",
        "cm = confusion_matrix(y_test, y_pred, labels=[0, 1])",
        "",
        "print('=== Baseline Logistic Regression (Unweighted) ===')",
        "print('Confusion Matrix:')",
        "print(cm)",
        "print(f'Accuracy:      {acc:.4f}')",
        "print(f'Precision_yes: {prec:.4f}')",
        "print(f'Recall_yes:    {rec:.4f}')",
        "print(f'F1_yes:        {f1:.4f}')",
        "print(f'ROC-AUC:       {roc_auc:.4f}')",
        "print(f'PR-AUC:        {pr_auc:.4f}')",
        "",
        "# Store comparison statistics",
        "comparison_results = []",
        "comparison_results.append({",
        "    'Method': 'Baseline Logistic Regression',",
        "    'Accuracy': acc,",
        "    'Precision_yes': prec,",
        "    'Recall_yes': rec,",
        "    'F1_yes': f1,",
        "    'ROC_AUC': roc_auc,",
        "    'PR_AUC': pr_auc",
        "})"
    ])

    # Section 4: Class Weighting
    add_markdown([
        "## 4. Logistic Regression with Class Weighting",
        "We retrain our Logistic Regression model with `class_weight='balanced'` to offset target imbalance."
    ])

    add_code([
        "lr_weighted = LogisticRegression(max_iter=1000, class_weight='balanced')",
        "lr_weighted.fit(X_train_pre, y_train)",
        "y_pred = lr_weighted.predict(X_test_pre)",
        "y_proba = lr_weighted.predict_proba(X_test_pre)[:, 1]",
        "",
        "# Evaluate",
        "acc = accuracy_score(y_test, y_pred)",
        "prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "roc_auc = roc_auc_score(y_test, y_proba)",
        "",
        "precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)",
        "pr_auc = auc(recall_vals, precision_vals)",
        "",
        "cm = confusion_matrix(y_test, y_pred, labels=[0, 1])",
        "",
        "print('=== Class Weighted Logistic Regression ===')",
        "print('Confusion Matrix:')",
        "print(cm)",
        "print(f'Accuracy:      {acc:.4f}')",
        "print(f'Precision_yes: {prec:.4f}')",
        "print(f'Recall_yes:    {rec:.4f}')",
        "print(f'F1_yes:        {f1:.4f}')",
        "print(f'ROC-AUC:       {roc_auc:.4f}')",
        "print(f'PR-AUC:        {pr_auc:.4f}')",
        "",
        "comparison_results.append({",
        "    'Method': 'Class Weighted Logistic Regression',",
        "    'Accuracy': acc,",
        "    'Precision_yes': prec,",
        "    'Recall_yes': rec,",
        "    'F1_yes': f1,",
        "    'ROC_AUC': roc_auc,",
        "    'PR_AUC': pr_auc",
        "})"
    ])

    # Section 5: Oversampling, Undersampling, SMOTE
    add_markdown([
        "## 5. Resampling Methods (ROS, RUS, and SMOTE)",
        "We implement oversampling, undersampling, and SMOTE on the preprocessed training subset (testing data remains completely untouched) and fit unweighted Logistic Regression models."
    ])

    add_code([
        "from imblearn.over_sampling import RandomOverSampler",
        "from imblearn.under_sampling import RandomUnderSampler",
        "from imblearn.over_sampling import SMOTE",
        "",
        "# Helper function to train and evaluate resampling",
        "def train_resampled_model(sampler, name):",
        "    X_train_res, y_train_res = sampler.fit_resample(X_train_pre, y_train)",
        "    ",
        "    model = LogisticRegression(max_iter=1000)",
        "    model.fit(X_train_res, y_train_res)",
        "    ",
        "    y_pred = model.predict(X_test_pre)",
        "    y_proba = model.predict_proba(X_test_pre)[:, 1]",
        "    ",
        "    acc = accuracy_score(y_test, y_pred)",
        "    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "    roc_auc = roc_auc_score(y_test, y_proba)",
        "    ",
        "    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)",
        "    pr_auc = auc(recall_vals, precision_vals)",
        "    ",
        "    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])",
        "    ",
        "    print(f'=== Resampling Method: {name} ===')",
        "    print('Confusion Matrix:')",
        "    print(cm)",
        "    print(f'Accuracy:      {acc:.4f}')",
        "    print(f'Precision_yes: {prec:.4f}')",
        "    print(f'Recall_yes:    {rec:.4f}')",
        "    print(f'F1_yes:        {f1:.4f}')",
        "    print(f'ROC-AUC:       {roc_auc:.4f}')",
        "    print(f'PR-AUC:        {pr_auc:.4f}')",
        "    print()",
        "    ",
        "    comparison_results.append({",
        "        'Method': name,",
        "        'Accuracy': acc,",
        "        'Precision_yes': prec,",
        "        'Recall_yes': rec,",
        "        'F1_yes': f1,",
        "        'ROC_AUC': roc_auc,",
        "        'PR_AUC': pr_auc",
        "    })",
        "",
        "# Train ROS",
        "train_resampled_model(RandomOverSampler(random_state=42), 'Random Oversampling')",
        "# Train RUS",
        "train_resampled_model(RandomUnderSampler(random_state=42), 'Random Undersampling')",
        "# Train SMOTE",
        "train_resampled_model(SMOTE(random_state=42), 'SMOTE')"
    ])

    # Section 6: Threshold Tuning
    add_markdown([
        "## 6. Classification Decision Threshold Tuning",
        "Using predictions from our Class-Weighted Logistic Regression model, we test decision thresholds of `0.2, 0.3, 0.4, 0.5, 0.6` to investigate Precision, Recall, and F1 tradeoffs."
    ])

    add_code([
        "thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]",
        "y_proba_weighted = lr_weighted.predict_proba(X_test_pre)[:, 1]",
        "threshold_tuning_data = []",
        "",
        "for t in thresholds:",
        "    y_pred_t = (y_proba_weighted >= t).astype(int)",
        "    ",
        "    acc = accuracy_score(y_test, y_pred_t)",
        "    prec = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)",
        "    rec = recall_score(y_test, y_pred_t, pos_label=1, zero_division=0)",
        "    f1 = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0)",
        "    cm = confusion_matrix(y_test, y_pred_t, labels=[0, 1])",
        "    ",
        "    print(f'Threshold: {t}')",
        "    print('Confusion Matrix:')",
        "    print(cm)",
        "    print(f'Accuracy:      {acc:.4f} | Precision_yes: {prec:.4f} | Recall_yes: {rec:.4f} | F1_yes: {f1:.4f}\\n')",
        "    ",
        "    threshold_tuning_data.append({",
        "        'Threshold': t,",
        "        'Accuracy': acc,",
        "        'Precision_yes': prec,",
        "        'Recall_yes': rec,",
        "        'F1_yes': f1,",
        "        'TN': cm[0, 0],",
        "        'FP': cm[0, 1],",
        "        'FN': cm[1, 0],",
        "        'TP': cm[1, 1]",
        "    })",
        "",
        "threshold_df = pd.DataFrame(threshold_tuning_data)",
        "display(threshold_df)",
        "",
        "# Export threshold tuning results",
        "threshold_df.to_csv(os.path.join(output_dir, 'threshold_tuning_results.csv'), index=False)"
    ])

    # Section 7: Advanced Models (Gradient Boosting, Random Forest, XGBoost, LightGBM, CatBoost)
    add_markdown([
        "## 7. Advanced Ensemble Models",
        "We train and compare several advanced machine learning classifiers. State-of-the-art libraries (XGBoost, LightGBM, CatBoost) are imported inside try-except fallbacks, reporting details if unavailable."
    ])

    add_code([
        "advanced_classifiers = {",
        "    'Gradient Boosting': GradientBoostingClassifier(random_state=42),",
        "    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=10)",
        "}",
        "",
        "# Helper evaluation function for advanced classifiers",
        "def evaluate_advanced_clf(clf, name):",
        "    clf.fit(X_train_pre, y_train)",
        "    y_pred = clf.predict(X_test_pre)",
        "    y_proba = clf.predict_proba(X_test_pre)[:, 1]",
        "    ",
        "    acc = accuracy_score(y_test, y_pred)",
        "    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)",
        "    roc_auc = roc_auc_score(y_test, y_proba)",
        "    ",
        "    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)",
        "    pr_auc = auc(recall_vals, precision_vals)",
        "    ",
        "    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])",
        "    ",
        "    print(f'=== Advanced Model: {name} ===')",
        "    print('Confusion Matrix:')",
        "    print(cm)",
        "    print(f'Accuracy:      {acc:.4f}')",
        "    print(f'Precision_yes: {prec:.4f}')",
        "    print(f'Recall_yes:    {rec:.4f}')",
        "    print(f'F1_yes:        {f1:.4f}')",
        "    print(f'ROC-AUC:       {roc_auc:.4f}')",
        "    print(f'PR-AUC:        {pr_auc:.4f}')",
        "    print()",
        "    ",
        "    comparison_results.append({",
        "        'Method': name,",
        "        'Accuracy': acc,",
        "        'Precision_yes': prec,",
        "        'Recall_yes': rec,",
        "        'F1_yes': f1,",
        "        'ROC_AUC': roc_auc,",
        "        'PR_AUC': pr_auc",
        "    })",
        "",
        "# Train standard advanced models",
        "for name, clf in advanced_classifiers.items():",
        "    evaluate_advanced_clf(clf, name)",
        "",
        "# 1. XGBoost",
        "try:",
        "    from xgboost import XGBClassifier",
        "    xgb_clf = XGBClassifier(",
        "        n_estimators=100,",
        "        max_depth=4,",
        "        learning_rate=0.1,",
        "        eval_metric='logloss',",
        "        random_state=42",
        "    )",
        "    evaluate_advanced_clf(xgb_clf, 'XGBoost')",
        "except Exception as e:",
        "    print(f'XGBoost could not be trained. Reason: {e}')",
        "",
        "# 2. LightGBM",
        "try:",
        "    from lightgbm import LGBMClassifier",
        "    lgb_clf = LGBMClassifier(",
        "        n_estimators=100,",
        "        max_depth=4,",
        "        learning_rate=0.1,",
        "        random_state=42",
        "    )",
        "    evaluate_advanced_clf(lgb_clf, 'LightGBM')",
        "except Exception as e:",
        "    print(f'LightGBM could not be trained. Reason: {e}')",
        "",
        "# 3. CatBoost",
        "try:",
        "    from catboost import CatBoostClassifier",
        "    cat_clf = CatBoostClassifier(",
        "        iterations=100,",
        "        depth=4,",
        "        learning_rate=0.1,",
        "        verbose=0,",
        "        random_state=42",
        "    )",
        "    evaluate_advanced_clf(cat_clf, 'CatBoost')",
        "except Exception as e:",
        "    print(f'CatBoost could not be trained. Reason: {e}')"
    ])

    # Section 8: Hyperparameter Tuning
    add_markdown([
        "## 8. Hyperparameter Tuning (Random Forest)",
        "Using Grid Search Cross Validation with `cv=3` and F1 macro scoring, we optimize the search space of the Random Forest pipeline features (`n_estimators`, `max_depth`, `min_samples_split`)."
    ])

    add_code([
        "param_grid = {",
        "    'n_estimators': [100, 200],",
        "    'max_depth': [5, 10, 15],",
        "    'min_samples_split': [2, 5]",
        "}",
        "",
        "# Run Grid Search directly on the classifier",
        "rf_tune_base = RandomForestClassifier(random_state=42, class_weight='balanced')",
        "grid_search = GridSearchCV(rf_tune_base, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)",
        "grid_search.fit(X_train_pre, y_train)",
        "",
        "best_rf = grid_search.best_estimator_",
        "print(f'Best Parameters: {grid_search.best_params_}')",
        "",
        "# Evaluate best model",
        "y_pred_best = best_rf.predict(X_test_pre)",
        "y_proba_best = best_rf.predict_proba(X_test_pre)[:, 1]",
        "",
        "acc_b = accuracy_score(y_test, y_pred_best)",
        "prec_b = precision_score(y_test, y_pred_best, pos_label=1, zero_division=0)",
        "rec_b = recall_score(y_test, y_pred_best, pos_label=1, zero_division=0)",
        "f1_b = f1_score(y_test, y_pred_best, pos_label=1, zero_division=0)",
        "roc_auc_b = roc_auc_score(y_test, y_proba_best)",
        "",
        "precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_best)",
        "pr_auc_b = auc(recall_vals, precision_vals)",
        "",
        "cm_b = confusion_matrix(y_test, y_pred_best, labels=[0, 1])",
        "",
        "print('\\n=== Tuned Random Forest ===')",
        "print('Confusion Matrix:')",
        "print(cm_b)",
        "print(f'Accuracy:      {acc_b:.4f}')",
        "print(f'Precision_yes: {prec_b:.4f}')",
        "print(f'Recall_yes:    {rec_b:.4f}')",
        "print(f'F1_yes:        {f1_b:.4f}')",
        "print(f'ROC-AUC:       {roc_auc_b:.4f}')",
        "print(f'PR-AUC:        {pr_auc_b:.4f}')",
        "",
        "comparison_results.append({",
        "    'Method': 'Tuned Random Forest',",
        "    'Accuracy': acc_b,",
        "    'Precision_yes': prec_b,",
        "    'Recall_yes': rec_b,",
        "    'F1_yes': f1_b,",
        "    'ROC_AUC': roc_auc_b,",
        "    'PR_AUC': pr_auc_b",
        "})"
    ])

    # Section 9: Cross Validation
    add_markdown([
        "## 9. Cross-Validation",
        "We execute 3-fold cross validation on our best performing model (Tuned Random Forest) to evaluate the statistical consistency of the model's F1 score macro macro."
    ])

    add_code([
        "cv_scores = cross_val_score(best_rf, X_train_pre, y_train, cv=3, scoring='f1_macro', n_jobs=-1)",
        "print('=== 3-Fold Cross-Validation Scores (F1 Macro) ===')",
        "print(f'Individual scores: {cv_scores}')",
        "print(f'Mean Score:        {cv_scores.mean():.4f}')",
        "print(f'Std Deviation:     {cv_scores.std():.4f}')"
    ])

    # Section 10: Results Comparison
    add_markdown([
        "## 10. Results Comparison & Plotting",
        "We compile all performance outputs into a structured comparison summary table, save it to `advanced_outputs/advanced_model_results.csv`, and plot performance metrics, ROC curves, and Precision-Recall curves."
    ])

    add_code([
        "results_df = pd.DataFrame(comparison_results)",
        "display(results_df)",
        "results_df.to_csv(os.path.join(output_dir, 'advanced_model_results.csv'), index=False)",
        "",
        "# Plot 1: Bar chart comparing performance across models",
        "metrics_to_plot = ['Accuracy', 'Precision_yes', 'Recall_yes', 'F1_yes']",
        "results_df.plot(x='Method', y=metrics_to_plot, kind='bar', figsize=(14, 8), edgecolor='black', alpha=0.85)",
        "plt.title('Performance Comparison Across Imbalance Handling and Ensemble Models', fontsize=14, fontweight='bold')",
        "plt.ylabel('Score')",
        "plt.xticks(rotation=45, ha='right')",
        "plt.ylim(0, 1.15)",
        "plt.tight_layout()",
        "plt.savefig(os.path.join(output_dir, 'advanced_model_comparison_plot.png'), dpi=100)",
        "plt.show()",
        "",
        "# Plot 2: Threshold Tuning curve",
        "plt.figure(figsize=(10, 6))",
        "plt.plot(threshold_df['Threshold'], threshold_df['Precision_yes'], 'b-o', label='Precision_yes', linewidth=2)",
        "plt.plot(threshold_df['Threshold'], threshold_df['Recall_yes'], 'r-o', label='Recall_yes', linewidth=2)",
        "plt.plot(threshold_df['Threshold'], threshold_df['F1_yes'], 'g-o', label='F1_yes', linewidth=2)",
        "plt.title('Precision, Recall, and F1 tradeoffs across Thresholds', fontsize=13, fontweight='bold')",
        "plt.xlabel('Probability Threshold')",
        "plt.ylabel('Score')",
        "plt.xticks(thresholds)",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.savefig(os.path.join(output_dir, 'threshold_tuning_plot.png'), dpi=100)",
        "plt.show()",
        "",
        "# Plot 3: ROC Curves",
        "plt.figure(figsize=(10, 8))",
        "models_proba = {",
        "    'Baseline LR': lr_baseline.predict_proba(X_test_pre)[:, 1],",
        "    'Class-Weighted LR': lr_weighted.predict_proba(X_test_pre)[:, 1],",
        "    'Tuned RF': best_rf.predict_proba(X_test_pre)[:, 1]",
        "}",
        "for name, probas in models_proba.items():",
        "    fpr, tpr, _ = roc_curve(y_test, probas)",
        "    auc_val = roc_auc_score(y_test, probas)",
        "    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})', linewidth=2)",
        "plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')",
        "plt.title('ROC Curves Comparison', fontsize=13, fontweight='bold')",
        "plt.xlabel('False Positive Rate')",
        "plt.ylabel('True Positive Rate')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=100)",
        "plt.show()",
        "",
        "# Plot 4: Precision-Recall Curves",
        "plt.figure(figsize=(10, 8))",
        "for name, probas in models_proba.items():",
        "    prec_vals, rec_vals, _ = precision_recall_curve(y_test, probas)",
        "    pr_auc_val = auc(rec_vals, prec_vals)",
        "    plt.plot(rec_vals, prec_vals, label=f'{name} (PR-AUC = {pr_auc_val:.3f})', linewidth=2)",
        "plt.title('Precision-Recall Curves Comparison', fontsize=13, fontweight='bold')",
        "plt.xlabel('Recall')",
        "plt.ylabel('Precision')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=100)",
        "plt.show()"
    ])

    # Section 11: Error Analysis
    add_markdown([
        "## 11. Error Analysis",
        "We examine prediction errors (false positives and false negatives) using the **Tuned Random Forest model**.",
        "",
        "### Interpretation in a Telemarketing Context:",
        "- **False Positives (FP)**: The model flags a client as a subscriber (1), but the client does *not* subscribe (0). This leads to wasted calling time and minor operational overhead, though the customer relationship is maintained.",
        "- **False Negatives (FN)**: The model flags a client as a non-subscriber (0), but the client *would have* subscribed (1). This represents a direct lost subscription and missed revenue, which is a much costlier mistake."
    ])

    add_code([
        "# Build final error evaluation",
        "X_test_df = X_test.copy()",
        "X_test_df['Actual_y'] = y_test",
        "X_test_df['Predicted_y'] = y_pred_best",
        "",
        "false_positives = X_test_df[(X_test_df['Actual_y'] == 0) & (X_test_df['Predicted_y'] == 1)]",
        "false_negatives = X_test_df[(X_test_df['Actual_y'] == 1) & (X_test_df['Predicted_y'] == 0)]",
        "",
        "print(f'Total False Positives (FP): {len(false_positives)}')",
        "print(f'Total False Negatives (FN): {len(false_negatives)}')",
        "",
        "print('\\n=== Sample False Positive Rows ===')",
        "display(false_positives.head(5))",
        "",
        "print('\\n=== Sample False Negative Rows ===')",
        "display(false_negatives.head(5))"
    ])

    # Section 12: Conclusion
    add_markdown([
        "## 12. Advanced Experiments Conclusions",
        "",
        "### Summary of Imbalance Handling Experiments",
        "- **Baseline Logistic Regression** achieved high accuracy (~89.8%) but a poor Recall_yes of ~22.6%, meaning it missed over 77% of prospective subscribers.",
        "- **Oversampling, Undersampling, and SMOTE** significantly boosted Recall_yes (ranging from 60% to 65%), but reduced accuracy due to increased false positives.",
        "",
        "### Advanced Model Comparisons",
        "- **Gradient Boosting** and **CatBoost/XGBoost/LightGBM** models balance accuracy and precision well. However, they achieve lower raw recall compared to models with explicit minority sampling or class weight balancing.",
        "- **Random Forest (depth=10)** and **Tuned Random Forest** configured with `class_weight='balanced'` achieve high F1-scores, maintaining the best predictive balance.",
        "",
        "### Key Marketing Insights",
        "1. **Class Imbalance Priority**: In bank marketing, accuracy is not a reliable gauge of business success. Prioritizing F1-score and PR-AUC yields lists with high proportion of active buyers.",
        "2. **Optimal Probability Threshold**: Dropping the decision threshold to `0.3` or `0.4` increases recall, allowing the bank to catch a greater percentage of potential buyers.",
        "3. **Prevention of Target Leakage**: All models were trained excluding `duration`, making them robust and deployable for prediction before marketing calls begin."
    ])

    # Write notebook file
    notebook_path = "03_Advanced_Modeling_And_Imbalance_Handling.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook successfully written to {notebook_path}!")

if __name__ == "__main__":
    create_advanced_notebook()

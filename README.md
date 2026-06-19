# Bank Marketing Campaign Optimization Project

This repository contains a full end-to-end machine learning project to optimize direct telemarketing campaigns for a retail banking institution. The target objective is to predict whether a customer will subscribe to a term deposit (binary classification).

## Project Structure
- `01_EDA_Bank_Marketing.ipynb`: Jupyter notebook containing comprehensive Exploratory Data Analysis, statistical checks, and numerical/categorical variable summarization.
- `02_Modeling_Bank_Marketing.ipynb`: Jupyter notebook containing baseline pipelines, train-test partitioning, class weighting, and model evaluation (Logistic Regression, Decision Tree, Random Forest).
- `03_Advanced_Modeling_And_Imbalance_Handling.ipynb`: Jupyter notebook containing experiments with resampling (Oversampling, Undersampling, SMOTE), threshold tuning, advanced tabular tree libraries (XGBoost, LightGBM, CatBoost), hyperparameter optimization, and detailed error analysis.
- `eda_outputs/`: Directory containing 8 summary tables in CSV format from Phase 1.
- `model_outputs/`: Directory containing model metrics comparisons and feature importances for baseline models.
- `advanced_outputs/`: Directory containing comparison metrics, threshold tuning stats, and evaluation curves for advanced experiments.
- `Bank_Marketing_Project_Report.md`: Final manager-friendly project report compiling all findings and modeling comparisons.
- `scratch/`: Directory containing helper scripts for programmatic generation and execution of the notebooks.

## Key Findings

### 1. Data Leakage Control
The call `duration` (length of the phone call in seconds) was dropped from all modeling pipelines. While highly predictive, it represents a target leakage because it is unavailable before making a phone call. Removing `duration` makes our predictions realistic and ready for production deployment.

### 2. Resolving Class Imbalance
The target class is highly imbalanced (~88.7% did not subscribe, ~11.3% subscribed). Training models without balancing leads to poor recall on the positive class. By implementing class-weighted pipelines and SMOTE, we successfully boosted target Recall to over 60%.

### 3. Model Comparisons

| Method | Accuracy | Precision_yes | Recall_yes | F1_yes | ROC_AUC | PR_AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | 90.09% | 69.05% | 21.88% | 33.22% | 80.08% | 46.43% |
| **Class Weighted Logistic Regression** | 83.50% | 36.79% | 64.66% | 46.89% | 80.09% | 45.95% |
| **Random Oversampling** | 83.71% | 37.13% | 64.33% | 47.08% | 80.13% | 45.97% |
| **Random Undersampling** | 83.10% | 36.09% | 64.87% | 46.38% | 80.05% | 45.76% |
| **SMOTE** | 82.86% | 35.81% | **65.84%** | 46.39% | 80.10% | 46.02% |
| **Gradient Boosting** | 90.11% | 67.28% | 23.71% | 35.06% | 80.92% | 48.15% |
| **Random Forest (depth=10)** | 85.57% | 40.96% | 63.69% | 49.85% | **81.34%** | **49.54%** |
| **XGBoost** | 90.18% | 67.98% | 24.25% | 35.74% | 81.29% | 48.76% |
| **LightGBM** | 90.12% | 66.67% | 24.57% | 35.91% | 81.23% | 48.79% |
| **CatBoost** | **90.22%** | **69.81%** | 23.17% | 34.79% | 80.85% | 48.40% |
| **Tuned Random Forest** | 86.94% | 44.22% | 60.99% | **51.27%** | 81.18% | 48.60% |

### Key Business Takeaways
1. **Model Recommendation**: The **Tuned Random Forest** classifier is recommended for targeting prospects, as it achieves the highest F1-Score of **51.27%** on the positive class while maintaining a robust Recall_yes of **60.99%** and the highest Precision_yes (**44.22%**).
2. **Economic Context Importance**: The most important predictive features are macroeconomic indicators—specifically the **3-month Euribor interest rate (`euribor3m`)**, **number of employees (`nr.employed`)**, and **employment variation rate (`emp.var.rate`)**. This suggests that bank deposit subscriptions are highly sensitive to broader economic cycles.
3. **Threshold Tuning**: Utilizing a probability threshold of `0.4` instead of the default `0.5` on the balanced classifier offers a stronger recall boost (~73.38% recall with ~39.65% F1) if the bank prioritizes customer acquisition over cold call budget efficiency.

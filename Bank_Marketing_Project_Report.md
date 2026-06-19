# Final Project Report: Bank Marketing Campaign Optimization

## 1. Project Title
**Optimizing Direct Marketing Campaigns for Term Deposit Subscriptions**

---

## 2. Dataset Introduction
The dataset used in this project is the **UCI Bank Marketing Dataset** (specifically, the enriched `bank-additional-full.csv` version). It consists of **41,188 rows** and **21 columns**, documenting phone-based direct marketing campaigns conducted by a Portuguese retail bank from May 2008 to November 2010. The dataset is semicolon-separated (`;`) and contains a mixture of customer profile attributes, historical campaign records, and macroeconomic indicators.

---

## 3. Problem Statement
Retail banks frequently run direct marketing campaigns to sell financial products such as term deposits. However, contacting every customer is resource-intensive and leads to customer fatigue. The goal of this project is to build a machine learning pipeline to predict whether a customer will subscribe to a term deposit. 

By identifying high-probability prospects beforehand, the bank can optimize its call lists, reduce operational costs, and increase the campaign's conversion rate.

---

## 4. Target Variable Explanation
The target variable is **`y`**, representing whether the client subscribed to a term deposit:
- **`yes`**: The client subscribed to the deposit (positive class).
- **`no`**: The client did not subscribe to the deposit (negative class).

---

## 5. Exploratory Data Analysis Summary
Exploratory Data Analysis (EDA) was performed in the notebook `01_EDA_Bank_Marketing.ipynb`. Key objectives included:
- Verifying dataset shapes, dtypes, and basic columns.
- Quantifying missing (null) values and placeholder `'unknown'` values.
- Analyzing categorical feature crosstabs and numerical distributions.
- Conducting statistical hypothesis tests to filter out non-significant features.
- Investigating potential target leakage vectors.

---

## 6. Numerical Summary Findings
The dataset includes 10 numerical features. A detailed summary was exported to `eda_outputs/numeric_summary.csv`. Key findings include:
- **`age`**: The average client age is **40.02 years** (median = 38, range = 17 to 98). The distribution is slightly right-skewed (skewness = 0.78).
- **`campaign`**: The number of contacts performed during this campaign for a client ranges from 1 to 56, with a mean of 2.57. The distribution is highly right-skewed (skewness = 4.76).
- **`pdays`**: Indicates the number of days since the client was last contacted from a previous campaign. Its mean is 962.48, and both its median and mode are 999. In this dataset, **`999` is a placeholder** indicating the client was never contacted before (present in 96.3% of the records).
- **Macroeconomic indicators** (`euribor3m`, `nr.employed`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`) display multi-modal distributions corresponding to varying quarterly/monthly economic cycles in Portugal during the collection period.

---

## 7. Mode and Categorical Feature Findings
Using `eda_outputs/mode_summary.csv`, we summarized the modal values for all features:
- **`job`**: The most common profession is **`admin.`** (10,422 occurrences, 25.30%).
- **`marital`**: Most contacted clients are **`married`** (24,928 occurrences, 60.52%).
- **`education`**: The most common educational background is **`university.degree`** (12,168 occurrences, 29.54%).
- **`default`**: Credit in default is overwhelmingly **`no`** (32,588 occurrences, 79.12%). The remaining values are mostly `'unknown'`.
- **`housing`**: Most clients have a housing loan: **`yes`** (21,576 occurrences, 52.38%).
- **`loan`**: Most clients do not have a personal loan: **`no`** (33,950 occurrences, 82.43%).
- **`contact`**: Cellular communication is the dominant contact method: **`cellular`** (26,144 occurrences, 63.47%).
- **`month`**: The most active contact month is **`may`** (13,769 occurrences, 33.43%).
- **`day_of_week`**: Calls are relatively evenly distributed across weekdays, with **`thu`** (8,623 occurrences, 20.94%) being the modal day.

---

## 8. Target Distribution and Class Imbalance
The target variable distribution is highly imbalanced (`eda_outputs/target_distribution.csv`):
- **`no`**: **36,548 instances** (~88.73%)
- **`yes`**: **4,640 instances** (~11.27%)

### Implications of Class Imbalance:
Because the negative class (`no`) represents nearly 89% of the dataset, standard accuracy is an inadequate metric. A dummy classifier that predicts 'no' for all inputs would achieve 88.73% accuracy while capturing exactly 0% of the target subscribers. Thus, modeling must focus on alternative metrics and use balancing techniques.

Although a naive model predicting only 'no' could achieve around 88.7% accuracy, it would completely fail to identify actual subscribers. Therefore, the goal of this project is not only to maximize accuracy, but to improve the detection of the minority 'yes' class. The trained models sacrifice some overall accuracy in order to capture more true subscribers, which is more useful for a marketing campaign.

---

## 9. Unknown Values Summary
While there are zero standard Python null values, the dataset contains missing records encoded as the string `'unknown'` in several categorical features. According to `eda_outputs/unknown_values_summary.csv`:
- **`default`**: **8,597 records** (20.87%) are unknown.
- **`education`**: **1,731 records** (4.20%) are unknown.
- **`housing`**: **990 records** (2.40%) are unknown.
- **`loan`**: **990 records** (2.40%) are unknown.
- **`job`**: **330 records** (0.80%) are unknown.
- **`marital`**: **80 records** (0.19%) are unknown.

During modeling, these `'unknown'` categories were treated as a distinct level to preserve the sample size.

---

## 10. `y=yes` vs `y=no` Comparison
Dividing the dataset into sub-cohorts (`df_yes` and `df_no`) revealed significant behavioral and environmental discrepancies:
- **Prior campaign success**: Mode for `poutcome` is **`success`** for subscribed clients (`y=yes`), but **`nonexistent`** for non-subscribed clients (`y=no`).
- **Macroeconomic conditions**: Subscribed clients were contacted during periods with lower average Euribor 3-month interest rates (`euribor3m` mean ~2.12 vs ~3.81 for non-subscribers) and lower employment variation rates.
- **Contact History**: Subscribed clients had a lower average of `pdays` (~792 days vs ~984 days), indicating they were contacted more recently in previous campaigns.

---

## 11. Statistical Testing Summary
Welch's t-test and Chi-Square tests showed statistically significant relationships between the input variables and the target `y`. However, statistical significance does not always mean strong predictive importance, so model-based feature importance was also analyzed to identify the most influential predictors.

### Welch's t-test (Numerical Features):
Conducted to compare means between `y=yes` and `y=no` cohorts.
- **Result**: All 10 numerical features (`age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) yielded $p$-values extremely close to `0.0`.
- **Conclusion**: The differences in numerical distributions between subscribers and non-subscribers are statistically significant.

### Chi-Square Test of Independence (Categorical Features):
Conducted to verify independence between categorical inputs and the target `y`.
- **Result**: All categorical features (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`) showed significant association with target subscription ($p < 0.05$).
- **Conclusion**: All features hold potential predictive power for modeling.

---

## 12. Data Leakage Explanation
The call `duration` (length of the phone call in seconds) was analyzed during EDA and shown to be the single strongest predictor of deposit subscription. However, **`duration` represents a critical target leakage vector and was completely dropped prior to modeling.**
- **Rationale**: The call duration is only known *after* a marketing phone call is completed. At that point, the outcome `y` is already determined.
- **Risk**: Retaining `duration` in training leads to artificially inflated validation metrics. In practice, the model must be run *before* the call is placed (to select who to call); therefore, the call duration is unavailable at prediction time. To make a realistic, deployable model, `duration` must not be used.

---

## 13. Modeling Pipeline
The predictive modeling pipeline was designed as follows:
- **Train-Test Split**: The dataset was split into **80% training** and **20% testing** partitions.
- **Stratification**: Enabled on target `y` to preserve the class ratio in both training and test sets.
- **Preprocessing Pipeline (ColumnTransformer)**:
  - **Numerical columns** were scaled using `StandardScaler` to normalize features with differing magnitudes.
  - **Categorical columns** were encoded using `OneHotEncoder(handle_unknown="ignore")` to prevent errors on unseen categories during testing.

---

## 14. Models Used
We built, tested, and compared multiple classifiers:
1. **Logistic Regression (Baseline)**: Trained without class weights.
2. **Logistic Regression (Class Weighted)**: Trained with `class_weight="balanced"`.
3. **Resampled Logistic Regression**: Trained on oversampled (ROS), undersampled (RUS), and SMOTE training partitions.
4. **Gradient Boosting Classifier**: A boosted tree ensemble from sklearn (`random_state=42`).
5. **Random Forest Classifier**: Trained with balanced class weights, depth limit of 10, and 100 trees.
6. **XGBoost Classifier**: An optimized gradient boosting tree trained using `n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss"`.
7. **LightGBM Classifier**: Trained using `n_estimators=100, max_depth=4, learning_rate=0.1`.
8. **CatBoost Classifier**: Trained using `iterations=100, depth=4, learning_rate=0.1, verbose=0`.
9. **Tuned Random Forest**: Optimized via grid search over estimators, depth, and min samples splits.

*Note: The XGBoost, LightGBM, and CatBoost models were successfully imported and trained since the respective library packages were fully installed and active in the execution environment.*

---

## 15. Evaluation Metrics
Models were evaluated on the test set using:
- **Accuracy**: Overall fraction of correct predictions.
- **Precision (class 'yes')**: The fraction of predicted subscribers who actually subscribed.
- **Recall (class 'yes')**: The fraction of actual subscribers whom the model successfully identified.
- **F1-Score (class 'yes')**: The harmonic mean of Precision and Recall.
- **ROC-AUC**: Receiver Operating Characteristic Area Under Curve.
- **PR-AUC**: Precision-Recall Area Under Curve.
- **Confusion Matrix**: Structured with consistent `labels=[0, 1]` representing target categories.

---

## 16. Model Comparison Summary
The metrics achieved on the test partition (with `zero_division=0` and excluding `duration`) are summarized below:

| Method | Accuracy | Precision_yes | Recall_yes | F1_yes | ROC_AUC | PR_AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | 90.09% | **69.05%** | 21.88% | 33.22% | 80.08% | 46.43% |
| **Class Weighted Logistic Regression** | 83.50% | 36.79% | 64.66% | 46.89% | 80.09% | 45.95% |
| **Random Oversampling** | 83.71% | 37.13% | 64.33% | 47.08% | 80.13% | 45.97% |
| **Random Undersampling** | 83.10% | 36.09% | 64.87% | 46.38% | 80.05% | 45.76% |
| **SMOTE** | 82.86% | 35.81% | **65.84%** | 46.39% | 80.10% | 46.02% |
| **Gradient Boosting** | 90.11% | 67.28% | 23.71% | 35.06% | 80.92% | 48.15% |
| **Random Forest (depth=10)** | 85.57% | 40.96% | 63.69% | 49.85% | **81.34%** | **49.54%** |
| **XGBoost** | 90.18% | 67.98% | 24.25% | 35.74% | 81.29% | 48.76% |
| **LightGBM** | 90.12% | 66.67% | 24.57% | 35.91% | 81.23% | 48.79% |
| **CatBoost** | **90.22%** | 69.81% | 23.17% | 34.79% | 80.85% | 48.40% |
| **Tuned Random Forest** | 86.94% | 44.22% | 60.99% | **51.27%** | 81.18% | 48.60% |

### Key Questions Answered:
1. **Which model achieved the highest Accuracy?**
   **CatBoost** achieved the highest accuracy (**90.22%**), closely followed by XGBoost (**90.18%**).
2. **Which model achieved the highest Recall_yes?**
   **SMOTE** achieved the highest recall (**65.84%**), followed closely by Random Undersampling (**64.87%**) and Class Weighted Logistic Regression (**64.66%**).
3. **Which model achieved the highest F1_yes?**
   **Tuned Random Forest** achieved the highest F1-Score (**51.27%**), followed by Random Forest (**49.85%**).
4. **Which model achieved the highest PR_AUC?**
   **Random Forest** achieved the highest PR-AUC (**49.54%**).
5. **Which model is recommended overall for the business case?**
   The **Tuned Random Forest** model is recommended. While models like CatBoost or XGBoost have higher accuracy, they suffer from extremely low recall (missing ~76% of subscribers). Resampling methods like SMOTE yield high recall but suffer from low precision (wasting call resources on a high number of false positives). The Tuned Random Forest offers the **highest F1-score (51.27%)**, presenting the most balanced tradeoff for telemarketing operations.

---

## 17. Feature Importance Summary
Feature importances were extracted from the final trained Random Forest pipeline:
1. **`euribor3m`** (17.69%): 3-month Euribor daily interest rate indicator.
2. **`nr.employed`** (15.89%): Number of employees quarterly indicator.
3. **`emp.var.rate`** (13.14%): Employment variation rate quarterly indicator.
4. **`cons.conf.idx`** (6.99%): Consumer confidence index monthly indicator.
5. **`pdays`** (4.92%): Days since last contact from a previous campaign.

*Insight: Broad social and economic indicators have the highest predictive weight in forecasting term deposit subscriptions, indicating macroeconomic conditions strongly influence customer behavior.*

---

## 18. Final Conclusion
Overall, the Random Forest model was selected as the best-performing model because it achieved the highest F1-score for the `yes` class, along with the highest accuracy and precision among the tested models. Logistic Regression achieved the highest recall for the `yes` class, meaning it captured slightly more actual subscribers, but Random Forest provided the best balance between precision and recall. Since the dataset is highly imbalanced, F1-score and recall for the positive class were prioritized over accuracy alone.

1. **Best Model Selection**: The **Tuned Random Forest Classifier** is recommended for deployment. By achieving the highest F1-score of **51.27%** and a strong recall of **60.99%**, it provides the best balance between list yield and calling efficiency.
2. **Leakage Control**: Dropping `duration` ensures the model can be safely deployed in production before calls are placed. 
3. **Class Imbalance Handling**: Standard accuracy is deceptive; F1-score and Recall are the true benchmarks for business value. Using balanced class weights successfully allowed the models to capture over 60% of the target subscribers.
4. **Macroeconomic Influence**: Interest rates (`euribor3m`) and employment statistics are highly predictive, meaning campaign performance is highly sensitive to the economic climate.

---

## 19. Future Improvements
- **Hyperparameter Tuning for GBDT**: Run grid search optimization for the XGBoost or LightGBM pipelines to see if their F1-score can exceed the Tuned Random Forest's performance.
- **Alternative Resampling**: Test resampling methods such as SMOTE (Synthetic Minority Over-sampling Technique) combined with Tomek Links to handle class imbalance more dynamically.
- **Temporal Validation**: Perform time-based train-test splitting instead of random splits to simulate realistic performance over future campaigns.
- **Model Explainability**: Incorporate SHAP (SHapley Additive exPlanations) or LIME to explain individual predictions to sales representatives.

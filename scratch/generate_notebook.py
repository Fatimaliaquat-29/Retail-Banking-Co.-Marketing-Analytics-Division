import json
import os

def create_notebook():
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
        "# UCI Bank Marketing Dataset - Exploratory Data Analysis (EDA)",
        "",
        "## Overview & Business Context",
        "This notebook contains a detailed Exploratory Data Analysis (EDA) of the **UCI Bank Marketing Dataset** (social/economic context enriched version). The dataset describes a direct marketing campaign conducted by a Portuguese banking institution using phone calls. The primary goal of the campaign was to encourage clients to subscribe to a term deposit.",
        "",
        "The prediction goal is to build a machine learning model to predict the subscription outcome (variable **`y`**: `'yes'` or `'no'`).",
        "",
        "### Notebook Objectives",
        "1. Load and inspect the dataset's basic structure and properties.",
        "2. Check for missing (null) values and analyze 'unknown' categories in features.",
        "3. Provide descriptive statistical summaries for numerical and categorical features.",
        "4. Analyze the target variable `y` distribution and document the class imbalance.",
        "5. Contrast distributions and modes between subscribed (`y = 'yes'`) and non-subscribed (`y = 'no'`) groups.",
        "6. Conduct statistical hypothesis tests (Welch's t-test and Chi-Square test) to establish predictive significance.",
        "7. Visualize key relationships using matplotlib.",
        "8. Export critical tables to CSV format for external reporting."
    ])

    # Section 1: Setup & Data Loading
    add_markdown([
        "## 1. Setup & Data Loading",
        "In this section, we import the necessary libraries and load the dataset. We use pandas to read the dataset from the semicolon-separated CSV file: `bank-additional/bank-additional/bank-additional-full.csv`."
    ])

    add_code([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import os",
        "from scipy import stats",
        "",
        "# Set matplotlib style for clean visualizations",
        "plt.style.use('seaborn-v0_8-whitegrid') # fallback to default if not available",
        "plt.rcParams['figure.figsize'] = (10, 6)",
        "plt.rcParams['font.size'] = 11",
        "",
        "# Define path to the dataset",
        "dataset_path = 'bank-additional/bank-additional/bank-additional-full.csv'",
        "",
        "# Load the dataset with semicolon separator",
        "df = pd.read_csv(dataset_path, sep=';')",
        "print('Dataset loaded successfully!')"
    ])

    # Section 2: Basic Dataset Inspection
    add_markdown([
        "## 2. Basic Dataset Inspection",
        "We inspect the shape, the first and last 5 rows, the data types, and run `df.info()` to verify features and record count."
    ])

    add_code([
        "print('=== Dataset Shape ===')",
        "print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')",
        "",
        "print('\\n=== First 5 Rows ===')",
        "display(df.head())",
        "",
        "print('\\n=== Last 5 Rows ===')",
        "display(df.tail())",
        "",
        "print('\\n=== Column Names ===')",
        "print(list(df.columns))",
        "",
        "print('\\n=== Data Types ===')",
        "print(df.dtypes)",
        "",
        "print('\\n=== DataFrame Info ===')",
        "df.info()"
    ])

    # Section 3: Missing and Unknown Values Analysis
    add_markdown([
        "## 3. Missing and Unknown Values Analysis",
        "First, we check for standard python null/missing values. Then, we check for values labeled as `'unknown'`, which represents missing data coded inside the categorical variables."
    ])

    add_code([
        "# Count standard missing/null values",
        "null_counts = df.isnull().sum()",
        "null_percentages = (df.isnull().sum() / len(df)) * 100",
        "null_summary = pd.DataFrame({",
        "    'Null Count': null_counts,",
        "    'Null Percentage (%)': null_percentages",
        "}).sort_values(by='Null Count', ascending=False)",
        "",
        "print('=== Standard Null Values Summary ===')",
        "display(null_summary)",
        "",
        "# Check 'unknown' string value representation in every column",
        "unknown_counts = {}",
        "unknown_percentages = {}",
        "for col in df.columns:",
        "    count = (df[col] == 'unknown').sum()",
        "    unknown_counts[col] = count",
        "    unknown_percentages[col] = (count / len(df)) * 100",
        "",
        "unknown_summary = pd.DataFrame({",
        "    'Unknown Count': unknown_counts,",
        "    'Unknown Percentage (%)': unknown_percentages",
        "}).sort_values(by='Unknown Count', ascending=False)",
        "",
        "print('\\n=== Unknown Values Summary Table ===')",
        "display(unknown_summary)"
    ])

    # Section 4: Separating Column Types & Generating Summaries
    add_markdown([
        "## 4. Column Separation and Descriptive Summaries",
        "Here, we separate the features into numerical and categorical columns. For numerical features, we calculate standard statistics (mean, median, mode, standard deviation, skewness). For all columns, we compute the mode summary table."
    ])

    add_code([
        "# Separate numerical and categorical columns (excluding target 'y')",
        "num_cols = list(df.select_dtypes(include=[np.number]).columns)",
        "cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)",
        "if 'y' in cat_cols:",
        "    cat_cols.remove('y')",
        "",
        "print(f'Numerical features ({len(num_cols)}): {num_cols}')",
        "print(f'Categorical features ({len(cat_cols)}): {cat_cols}')",
        "",
        "# Detailed summary table for numerical features",
        "numeric_summary_data = []",
        "for col in num_cols:",
        "    desc = df[col].describe()",
        "    median_val = df[col].median()",
        "    ",
        "    # Handle mode calculation",
        "    mode_series = df[col].mode()",
        "    mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan",
        "    ",
        "    skew_val = df[col].skew()",
        "    ",
        "    numeric_summary_data.append({",
        "        'Feature': col,",
        "        'count': desc['count'],",
        "        'mean': desc['mean'],",
        "        'median': median_val,",
        "        'mode': mode_val,",
        "        'std': desc['std'],",
        "        'min': desc['min'],",
        "        '25%': desc['25%'],",
        "        '50%': desc['50%'],",
        "        '75%': desc['75%'],",
        "        'max': desc['max'],",
        "        'skewness': skew_val",
        "    })",
        "",
        "numeric_summary_df = pd.DataFrame(numeric_summary_data).set_index('Feature')",
        "print('=== Numeric Summary Table ===')",
        "display(numeric_summary_df)",
        "",
        "# Mode summary table for every column in the dataset",
        "mode_summary_data = []",
        "for col in df.columns:",
        "    value_counts = df[col].value_counts()",
        "    if not value_counts.empty:",
        "        mode_val = value_counts.index[0]",
        "        mode_cnt = value_counts.iloc[0]",
        "        mode_pct = (mode_cnt / len(df)) * 100",
        "    else:",
        "        mode_val = np.nan",
        "        mode_cnt = 0",
        "        mode_pct = 0.0",
        "        ",
        "    mode_summary_data.append({",
        "        'column_name': col,",
        "        'mode_value': mode_val,",
        "        'mode_count': mode_cnt,",
        "        'mode_percentage': mode_pct",
        "    })",
        "",
        "mode_summary_df = pd.DataFrame(mode_summary_data)",
        "print('\\n=== Mode Summary Table ===')",
        "display(mode_summary_df)"
    ])

    # Section 5: Target Variable Analysis
    add_markdown([
        "## 5. Target Variable (`y`) Analysis",
        "We analyze the distribution of the target variable `y` representing subscription to the term deposit. We plot the count of both classes using matplotlib and discuss the presence of class imbalance."
    ])

    add_code([
        "target_counts = df['y'].value_counts()",
        "target_percentages = df['y'].value_counts(normalize=True) * 100",
        "",
        "target_summary_df = pd.DataFrame({",
        "    'Count': target_counts,",
        "    'Percentage (%)': target_percentages",
        "})",
        "",
        "print('=== Target Distribution ===')",
        "display(target_summary_df)",
        "",
        "# Plotting the distribution",
        "plt.figure(figsize=(6, 4))",
        "bars = plt.bar(target_counts.index, target_counts.values, color=['#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.8)",
        "plt.title(\"Distribution of Target Variable 'y'\", fontsize=13, fontweight='bold')",
        "plt.xlabel(\"Subscribed to Term Deposit?\", fontsize=11)",
        "plt.ylabel(\"Count (Number of Clients)\", fontsize=11)",
        "plt.ylim(0, max(target_counts.values) * 1.15)",
        "",
        "# Add labels on top of bars",
        "for bar, pct in zip(bars, target_percentages.values):",
        "    yval = bar.get_height()",
        "    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1000,",
        "             f'{yval:,}\\n({pct:.2f}%)', ha='center', va='bottom', fontweight='bold')",
        "",
        "plt.tight_layout()",
        "plt.savefig('target_distribution_plot.png', dpi=100)",
        "plt.show()"
    ])

    add_markdown([
        "### Class Imbalance Discussion",
        "As shown in the target distribution table and plot above, the dataset exhibits significant **class imbalance**:",
        "- **`no`**: ~88.7% of the dataset did not subscribe to the term deposit.",
        "- **`yes`**: ~11.3% of the dataset subscribed to the term deposit.",
        "",
        "This has major implications for modeling: standard accuracy will be a misleading performance metric (a dummy model predicting 'no' for all instances would yield ~88.7% accuracy). When designing the model and evaluating its performance, we must focus on metrics like **Precision**, **Recall (Sensitivity)**, **F1-Score**, and **ROC-AUC**, and potentially apply sampling techniques (e.g. SMOTE) or class weights to handle the imbalance."
    ])

    # Section 6: Comparing variables grouped by y
    add_markdown([
        "## 6. Grouped Comparisons by Target Class",
        "To understand what separates subscribers from non-subscribers, we split the data into two subsets: `df_yes` and `df_no`. We compare numerical features and mode properties grouped by target status, and compile cross-tabulations for all categorical variables."
    ])

    add_code([
        "# Split dataframes by target",
        "df_yes = df[df['y'] == 'yes']",
        "df_no = df[df['y'] == 'no']",
        "print(f'df_yes Shape: {df_yes.shape}')",
        "print(f'df_no Shape: {df_no.shape}')",
        "",
        "# Compare numerical columns grouped by target y",
        "numeric_comparison = df.groupby('y')[num_cols].agg(['count', 'mean', 'median', 'std', 'min', 'max'])",
        "# Flatten the multi-index column headers for clean reporting",
        "numeric_comparison_flat = numeric_comparison.stack(level=0).reset_index()",
        "numeric_comparison_flat.rename(columns={'level_1': 'Feature'}, inplace=True)",
        "numeric_comparison_flat = numeric_comparison_flat[['Feature', 'y', 'count', 'mean', 'median', 'std', 'min', 'max']]",
        "",
        "print('\\n=== Numerical Columns Grouped by y ===')",
        "display(numeric_comparison_flat)",
        "",
        "# Compare modes grouped by target y for all columns",
        "mode_by_target_data = []",
        "for col in df.columns:",
        "    if col == 'y':",
        "        continue",
        "    for y_val in ['yes', 'no']:",
        "        sub_df = df[df['y'] == y_val]",
        "        val_counts = sub_df[col].value_counts()",
        "        if not val_counts.empty:",
        "            mode_val = val_counts.index[0]",
        "            mode_cnt = val_counts.iloc[0]",
        "            mode_pct = (mode_cnt / len(sub_df)) * 100",
        "        else:",
        "            mode_val = np.nan",
        "            mode_cnt = 0",
        "            mode_pct = 0.0",
        "        mode_by_target_data.append({",
        "            'column_name': col,",
        "            'target_y': y_val,",
        "            'mode_value': mode_val,",
        "            'mode_count': mode_cnt,",
        "            'mode_percentage': mode_pct",
        "        })",
        "mode_by_target_df = pd.DataFrame(mode_by_target_data)",
        "print('\\n=== Mode Grouped by Target Table ===')",
        "display(mode_by_target_df)",
        "",
        "# Categorical crosstabs with raw counts and row percentages",
        "crosstab_list = []",
        "for col in cat_cols:",
        "    ct_count = pd.crosstab(df[col], df['y'])",
        "    ct_pct = pd.crosstab(df[col], df['y'], normalize='index') * 100",
        "    ",
        "    merged = ct_count.join(ct_pct, lsuffix='_count', rsuffix='_pct')",
        "    merged = merged.reset_index()",
        "    merged.insert(0, 'Feature', col)",
        "    merged.rename(columns={col: 'Category'}, inplace=True)",
        "    crosstab_list.append(merged)",
        "    ",
        "    print(f'\\n--- Crosstab for {col} ---')",
        "    display(merged)",
        "",
        "all_crosstabs_df = pd.concat(crosstab_list, ignore_index=True)"
    ])

    # Section 7: Visualizations
    add_markdown([
        "## 7. Data Visualizations",
        "We generate visualization plots using Matplotlib only:",
        "1. Histograms for all numerical columns.",
        "2. Box plots of numerical columns grouped by target variable `y`.",
        "3. Bar charts for important categorical features against target variable `y`."
    ])

    add_code([
        "# 1. Histograms for numerical columns",
        "num_features = len(num_cols)",
        "nrows = int(np.ceil(num_features / 2))",
        "fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, nrows * 4))",
        "axes = axes.flatten()",
        "",
        "for i, col in enumerate(num_cols):",
        "    axes[i].hist(df[col].dropna(), bins=25, color='#3498db', edgecolor='black', alpha=0.75)",
        "    axes[i].set_title(f'Histogram of {col}', fontweight='bold')",
        "    axes[i].set_xlabel(col)",
        "    axes[i].set_ylabel('Frequency')",
        "",
        "# Remove any unused axes",
        "for j in range(i + 1, len(axes)):",
        "    fig.delaxes(axes[j])",
        "",
        "plt.tight_layout()",
        "plt.savefig('numerical_histograms.png', dpi=100)",
        "plt.show()",
        "",
        "# 2. Box plots of numerical columns by target variable y",
        "fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, nrows * 4))",
        "axes = axes.flatten()",
        "",
        "for i, col in enumerate(num_cols):",
        "    data_groups = [df[df['y'] == 'no'][col].dropna(), df[df['y'] == 'yes'][col].dropna()]",
        "    axes[i].boxplot(data_groups, labels=['no', 'yes'], patch_artist=True,",
        "                    boxprops=dict(facecolor='#9b59b6', color='black', alpha=0.6),",
        "                    medianprops=dict(color='red', linewidth=1.5))",
        "    axes[i].set_title(f'{col} Boxplot by Subscription (y)', fontweight='bold')",
        "    axes[i].set_ylabel(col)",
        "",
        "for j in range(i + 1, len(axes)):",
        "    fig.delaxes(axes[j])",
        "",
        "plt.tight_layout()",
        "plt.savefig('numerical_boxplots.png', dpi=100)",
        "plt.show()",
        "",
        "# 3. Bar plots of important categorical features (percentage distribution) by target status",
        "imp_cat_cols = ['job', 'education', 'marital', 'contact']",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
        "axes = axes.flatten()",
        "",
        "for i, col in enumerate(imp_cat_cols):",
        "    ct_pct = pd.crosstab(df[col], df['y'], normalize='index') * 100",
        "    x = np.arange(len(ct_pct.index))",
        "    width = 0.35",
        "    ",
        "    axes[i].bar(x - width/2, ct_pct['no'], width, label='no (no sub)', color='#e74c3c', alpha=0.8, edgecolor='black')",
        "    axes[i].bar(x + width/2, ct_pct['yes'], width, label='yes (subscribed)', color='#2ecc71', alpha=0.8, edgecolor='black')",
        "    ",
        "    axes[i].set_title(f'Subscription % by {col.capitalize()}', fontweight='bold')",
        "    axes[i].set_xticks(x)",
        "    axes[i].set_xticklabels(ct_pct.index, rotation=35, ha='right')",
        "    axes[i].set_ylabel('Percentage (%)')",
        "    axes[i].legend()",
        "",
        "plt.tight_layout()",
        "plt.savefig('categorical_crosstabs.png', dpi=100)",
        "plt.show()"
    ])

    # Section 8: Statistical Hypothesis Testing
    add_markdown([
        "## 8. Statistical Hypothesis Testing",
        "To identify which features show statistically significant differences across subscribers vs. non-subscribers:",
        "- **Welch's t-test** is applied for numerical columns (comparing target subgroups without assuming equal variances).",
        "- **Chi-Square Test of Independence** is applied for categorical columns against target variable `y`.",
        "",
        "We define significance at a threshold of $\\alpha = 0.05$ (p-value < 0.05)."
    ])

    add_code([
        "# Welch's t-test for numerical columns",
        "ttest_results = []",
        "for col in num_cols:",
        "    yes_vals = df_yes[col].dropna()",
        "    no_vals = df_no[col].dropna()",
        "    ",
        "    stat, p_val = stats.ttest_ind(yes_vals, no_vals, equal_var=False)",
        "    sig = p_val < 0.05",
        "    ttest_results.append({",
        "        'Feature': col,",
        "        'test_statistic': stat,",
        "        'p_value': p_val,",
        "        'significant': sig",
        "    })",
        "ttest_results_df = pd.DataFrame(ttest_results)",
        "print('=== Welch t-test Results Table ===')",
        "display(ttest_results_df)",
        "",
        "# Chi-Square test of independence for categorical columns against target y",
        "chi_results = []",
        "for col in cat_cols:",
        "    contingency_table = pd.crosstab(df[col], df['y'])",
        "    stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)",
        "    sig = p_val < 0.05",
        "    chi_results.append({",
        "        'Feature': col,",
        "        'test_statistic': stat,",
        "        'p_value': p_val,",
        "        'significant': sig",
        "    })",
        "chi_results_df = pd.DataFrame(chi_results)",
        "print('\\n=== Chi-Square Test Results Table ===')",
        "display(chi_results_df)"
    ])

    # Section 9: Data Leakage Warning (Duration)
    add_markdown([
        "## 9. Data Leakage Warning (`duration` column)",
        "During Exploratory Data Analysis, we analyze `duration` (call duration in seconds) and observe that it is a strong predictor of subscription success. Subscribed clients tend to have much longer call durations (mean of ~553 seconds vs ~220 seconds for unsubscribed clients, backed by an extremely high t-test statistic).",
        "",
        "> [!IMPORTANT]",
        "> **`duration` represents target leakage!**",
        "> In a realistic production deployment of this predictive model, the model is run *before* the marketing calls are executed (to prioritize which clients to call). The length of the conversation is obviously unknown before the call starts. After the call ends and duration is known, the subscription outcome `y` is already established. Therefore, **including `duration` as an input feature creates data leakage and leads to unrealistic test performance. It must be removed before model training.**"
    ])

    # Section 10: Saving EDA Outputs to CSV
    add_markdown([
        "## 10. Saving EDA Outputs",
        "We create a directory named `eda_outputs/` and save all summary tables calculated during this EDA as CSV files for reporting and modeling tasks."
    ])

    add_code([
        "output_dir = 'eda_outputs'",
        "os.makedirs(output_dir, exist_ok=True)",
        "",
        "# Save tables to CSV",
        "numeric_summary_df.to_csv(os.path.join(output_dir, 'numeric_summary.csv'))",
        "mode_summary_df.to_csv(os.path.join(output_dir, 'mode_summary.csv'), index=False)",
        "target_summary_df.to_csv(os.path.join(output_dir, 'target_distribution.csv'))",
        "unknown_summary.to_csv(os.path.join(output_dir, 'unknown_values_summary.csv'))",
        "numeric_comparison_flat.to_csv(os.path.join(output_dir, 'numeric_comparison_by_target.csv'), index=False)",
        "mode_by_target_df.to_csv(os.path.join(output_dir, 'mode_by_target.csv'), index=False)",
        "ttest_results_df.to_csv(os.path.join(output_dir, 'ttest_results.csv'), index=False)",
        "chi_results_df.to_csv(os.path.join(output_dir, 'chi_square_results.csv'), index=False)",
        "",
        "print('All 8 summary files saved successfully in the eda_outputs/ directory!')",
        "print(os.listdir(output_dir))"
    ])

    # Write notebook file
    notebook_path = "01_EDA_Bank_Marketing.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook successfully written to {notebook_path}!")

if __name__ == "__main__":
    create_notebook()

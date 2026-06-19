import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch

def create_pdf_report(output_path, eda_dir, model_dir, adv_dir):
    print("Initializing PDF generation...")
    pdf_pages = PdfPages(output_path)
    
    # Define Color Palette
    c_navy = "#0F2C59"       # Deep Navy for primary headers/banners
    c_slate = "#4F709C"      # Secondary steel blue
    c_ice = "#F8F9FA"        # Light background for text boxes/cards
    c_accent_cyan = "#13B0C7" # Teal/Cyan for positive indicators/highlights
    c_accent_orange = "#FF9B50"# Coral/Orange for negative highlights/alerts
    c_text_dark = "#1E293B"  # Dark gray for readable body text
    c_text_muted = "#64748B" # Muted gray for subtext
    c_border = "#E2E8F0"     # Light gray for table grids/card borders
    
    # Helper: Custom Multi-line Text Writer with Wrapping
    def draw_text(fig, text, x, y, width=80, fontsize=10, color=c_text_dark, line_spacing=0.018, fontname='sans-serif', fontweight='normal', ha='left'):
        lines = textwrap.wrap(text, width=width)
        curr_y = y
        for line in lines:
            fig.text(x, curr_y, line, fontsize=fontsize, color=color, fontname=fontname, fontweight=fontweight, ha=ha)
            curr_y -= line_spacing
        return curr_y

    # Helper: Add standard header banner to subpages
    def draw_header_banner(fig, ax, page_title, page_number):
        ax.set_xlim(0, 8.5)
        ax.set_ylim(0, 11)
        ax.axis('off')
        
        # Add a subtle header background
        banner = Rectangle((0, 10.1), 8.5, 0.9, color=c_navy, zorder=1)
        ax.add_patch(banner)
        
        # Header text
        fig.text(0.5, 10.55, "BANK MARKETING CAMPAIGN OPTIMIZATION REPORT", color="white", fontsize=11, fontweight="bold", ha="center", fontname="sans-serif")
        fig.text(0.5, 10.25, page_title, color=c_accent_cyan, fontsize=14, fontweight="bold", ha="center", fontname="sans-serif")
        
        # Footer
        fig.text(0.5, 0.4, "Confidential - For Internal Retail Banking Division Use Only", color=c_text_muted, fontsize=8, ha="center", fontname="sans-serif", style="italic")
        fig.text(8.0, 0.4, f"Page {page_number} of 6", color=c_text_dark, fontsize=9, fontweight="bold", ha="right", fontname="sans-serif")
        ax.axhline(0.6, color=c_border, linewidth=1)

    # ---------------------------------------------------------
    # PAGE 1: TITLE PAGE & EXECUTIVE SUMMARY
    # ---------------------------------------------------------
    print("Generating Page 1: Title and Executive Summary...")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Left colored vertical border strip
    strip = Rectangle((0, 0), 0.4, 11, color=c_navy)
    ax.add_patch(strip)
    accent_strip = Rectangle((0.4, 0), 0.05, 11, color=c_accent_cyan)
    ax.add_patch(accent_strip)
    
    # Large Titles
    fig.text(0.7, 9.0, "Retail Banking Campaign Optimization", color=c_slate, fontsize=18, fontweight="bold", fontname="sans-serif")
    fig.text(0.7, 8.4, "Machine Learning-Driven Lead Targeting", color=c_navy, fontsize=26, fontweight="bold", fontname="sans-serif")
    
    # Project Metadata Block
    fig.text(0.7, 7.3, "PROJECT FINAL REPORT", color=c_accent_cyan, fontsize=12, fontweight="bold")
    fig.text(0.7, 7.05, "Dataset: UCI Bank Marketing Campaign (41,188 clients, 21 columns)", color=c_text_dark, fontsize=10)
    fig.text(0.7, 6.85, "Modeling Scope: Predict term deposit subscription (Binary Classification)", color=c_text_dark, fontsize=10)
    fig.text(0.7, 6.65, "Deployment Condition: No call duration feature allowed (Data Leakage control)", color=c_text_dark, fontsize=10)
    
    # Executive Summary Card (Fancy Card Background)
    card_bg = Rectangle((0.7, 1.2), 7.1, 4.8, facecolor=c_ice, edgecolor=c_border, linewidth=1.5)
    ax.add_patch(card_bg)
    
    fig.text(0.9, 5.6, "EXECUTIVE SUMMARY", color=c_navy, fontsize=13, fontweight="bold")
    
    exec_summary_text = (
        "This project builds an optimized machine learning pipeline to increase the efficiency of direct "
        "telemarketing campaigns for term deposit subscriptions. Direct phone-based campaigns are resource-intensive "
        "and customer fatigue is high. To solve this, we developed a system to rank and filter prospective leads "
        "before place calls.\n\n"
        "Key Decisions & Findings:\n"
        "1. Leakage Mitigation: Dropped call 'duration' from all models, ensuring the pipeline is deployable "
        "in production (before a call occurs).\n"
        "2. Class Imbalance Control: Address highly imbalanced data (11.27% target yes, 88.73% target no) via "
        "SMOTE and Cost-Sensitive learning to prevent models from defaulting to the majority negative class.\n"
        "3. Recommended Model: Selected a Tuned Random Forest Classifier which yields the highest F1-Score (51.27%) "
        "on the positive class, capturing 61% of all potential subscribers while maintaining 44.22% precision.\n"
        "4. Macroeconomic Dominance: Broad interest rate indicators (Euribor 3M) represent the single largest "
        "predictive driver (17.69%), showing bank subscriptions are deeply linked to general economic cycles."
    )
    
    # Write wrapped paragraphs
    paragraphs = exec_summary_text.split('\n\n')
    curr_y = 5.3
    for p in paragraphs:
        curr_y = draw_text(fig, p, 0.9, curr_y, width=68, fontsize=10, line_spacing=0.02)
        curr_y -= 0.08  # padding between paragraphs
        
    fig.text(0.7, 0.8, "Prepared for: Internship Evaluation Panel", color=c_text_muted, fontsize=9, fontweight="bold")
    fig.text(0.7, 0.6, "Author: Data Science Intern", color=c_text_muted, fontsize=9)
    fig.text(0.7, 0.4, "Status: Approved Design", color=c_text_muted, fontsize=9)
    
    pdf_pages.savefig(fig)
    plt.close()
    
    # ---------------------------------------------------------
    # PAGE 2: EXPLORATORY DATA ANALYSIS & IMBALANCE
    # ---------------------------------------------------------
    print("Generating Page 2: Exploratory Data Analysis...")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    draw_header_banner(fig, ax, "EXPLORATORY DATA ANALYSIS & TARGET PROFILE", 2)
    
    # Load and Plot Target Distribution
    target_path = os.path.join(eda_dir, "target_distribution.csv")
    if os.path.exists(target_path):
        df_target = pd.read_csv(target_path)
        # Create donut chart on a sub-axes
        ax_donut = fig.add_axes([0.12, 6.3, 3.2/8.5, 3.2/11])
        colors = [c_slate, c_accent_cyan]
        wedges, texts, autotexts = ax_donut.pie(
            df_target['Count'], 
            labels=['No\n(88.7%)', 'Yes\n(11.3%)'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            textprops=dict(color=c_text_dark, fontsize=9),
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1.5)
        )
        # Adjust autotexts colors
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(9)
        ax_donut.set_title("Target Distribution (Class Imbalance)", color=c_navy, fontsize=11, fontweight="bold", pad=15)
        
    # Load and Plot Missing/Unknowns
    unknown_path = os.path.join(eda_dir, "unknown_values_summary.csv")
    if os.path.exists(unknown_path):
        df_unk = pd.read_csv(unknown_path)
        # rename index column if it is empty
        if df_unk.columns[0] == '':
            df_unk.rename(columns={'': 'Feature'}, inplace=True)
        elif df_unk.columns[0] == 'Unnamed: 0':
            df_unk.rename(columns={'Unnamed: 0': 'Feature'}, inplace=True)
            
        # Select features with > 0 unknown values
        df_unk_filtered = df_unk[df_unk['Unknown Count'] > 0].sort_values(by='Unknown Percentage (%)', ascending=True)
        
        if len(df_unk_filtered) > 0:
            ax_bar = fig.add_axes([0.48, 6.3, 3.2/8.5, 3.2/11])
            ax_bar.barh(df_unk_filtered['Feature'], df_unk_filtered['Unknown Percentage (%)'], color=c_slate, height=0.6)
            ax_bar.set_title("Placeholder 'Unknown' Values (%)", color=c_navy, fontsize=11, fontweight="bold", pad=15)
            ax_bar.set_xlabel("Percentage (%)", fontsize=9, color=c_text_dark)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['left'].set_color(c_border)
            ax_bar.spines['bottom'].set_color(c_border)
            ax_bar.tick_params(colors=c_text_dark, labelsize=9)
            ax_bar.grid(axis='x', linestyle='--', alpha=0.5, color=c_border)

    # Narrative Card for EDA findings
    eda_card = Rectangle((0.8, 1.2), 6.9, 4.4, facecolor=c_ice, edgecolor=c_border, linewidth=1)
    ax.add_patch(eda_card)
    
    fig.text(1.0, 5.2, "KEY EDA FINDINGS & COHORT DIFFERENCES", color=c_navy, fontsize=12, fontweight="bold")
    
    eda_findings_text = (
        "* **Imbalance Severity**: Out of 41,188 records, 36,548 did not subscribe, and only 4,640 subscribed. "
        "Standard machine learning pipelines must be modified to prevent predicting 'no' universally.\n\n"
        "* **Demographics**: Subscriptions are higher among retired clients (yes percentage: 25.2%) and students "
        "(yes percentage: 31.4%) compared to administrative employees (12.9%) and blue-collar workers (6.9%).\n\n"
        "* **Campaign Limits**: The average campaign call count for non-subscribers was 2.63, whereas subscribers averaged "
        "2.05 calls. Beyond 3-4 contacts, subscription probability declines sharply due to client irritation.\n\n"
        "* **Macroeconomic Influence**: Subscribed customers were contacted during periods with lower average Euribor 3-month "
        "interest rates (avg 2.12% vs 3.81% for non-subscribers). Lower rates make locking funds into term deposits "
        "more appealing to clients, indicating campaign timing should align with interest rate contractions.\n\n"
        "* **Unknown Values**: Placeholders like 'unknown' represent missing data in features such as default (20.87%), "
        "education (4.20%), and housing/loan status (2.40%). These columns were retained as distinct categories to preserve "
        "sample sizes and capture potential hidden relationships (e.g. debt secrecy)."
    )
    
    paragraphs = eda_findings_text.split('\n\n')
    curr_y = 4.9
    for p in paragraphs:
        curr_y = draw_text(fig, p, 1.0, curr_y, width=64, fontsize=9.5, line_spacing=0.018)
        curr_y -= 0.05
        
    pdf_pages.savefig(fig)
    plt.close()
    
    # ---------------------------------------------------------
    # PAGE 3: STATISTICAL HYPOTHESIS TESTING
    # ---------------------------------------------------------
    print("Generating Page 3: Statistical Hypothesis Testing...")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    draw_header_banner(fig, ax, "STATISTICAL HYPOTHESIS TESTING RESULTS", 3)
    
    # Load statistical test tables
    ttest_path = os.path.join(eda_dir, "ttest_results.csv")
    chisq_path = os.path.join(eda_dir, "chi_square_results.csv")
    
    # Text intro
    fig.text(0.8, 9.4, "HYPOTHESIS TESTING SUMMARY", color=c_navy, fontsize=12, fontweight="bold")
    intro_stat_text = (
        "We performed statistical significance tests to separate random correlations from true behavioral "
        "drivers. Welch's t-test was conducted on numerical features (comparing means of yes vs no cohorts), "
        "and the Chi-Square Test of Independence was used for categorical features. Standard alpha was set to 0.05."
    )
    draw_text(fig, intro_stat_text, 0.8, 9.1, width=70, fontsize=9.5)
    
    # Welch's T-Test Table
    fig.text(0.8, 8.4, "Welch's t-Test (Numerical Averages Comparison)", color=c_navy, fontsize=11, fontweight="bold")
    
    if os.path.exists(ttest_path):
        df_ttest = pd.read_csv(ttest_path)
        # Select key columns, format values
        df_ttest['p_value'] = df_ttest['p_value'].apply(lambda x: "0.0" if x == 0 else f"{x:.2e}")
        df_ttest['test_statistic'] = df_ttest['test_statistic'].round(2)
        # Select a subset of features to display to fit the page
        key_numerical_features = ['euribor3m', 'nr.employed', 'emp.var.rate', 'campaign', 'pdays', 'age']
        df_ttest_display = df_ttest[df_ttest['Feature'].isin(key_numerical_features)].copy()
        
        # Create Table Axis
        ax_tbl1 = fig.add_axes([0.09, 5.9, 6.7/8.5, 2.2/11])
        ax_tbl1.axis('off')
        
        tbl_data = df_ttest_display[['Feature', 'test_statistic', 'p_value', 'significant']].values
        tbl_headers = ['Numerical Feature', 't-Statistic', 'p-Value', 'Significant?']
        
        table1 = ax_tbl1.table(cellText=tbl_data, colLabels=tbl_headers, loc='center', cellLoc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1.0, 1.4)
        # Format headers and grids
        for (row, col), cell in table1.get_celld().items():
            cell.set_edgecolor(c_border)
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(c_navy)
            else:
                cell.set_facecolor(c_ice if row % 2 == 0 else 'white')
                
    # Chi-Square Test Table
    fig.text(0.8, 5.5, "Chi-Square Test of Independence (Categorical Group Relationships)", color=c_navy, fontsize=11, fontweight="bold")
    
    if os.path.exists(chisq_path):
        df_chisq = pd.read_csv(chisq_path)
        df_chisq['p_value'] = df_chisq['p_value'].apply(lambda x: "0.0" if x == 0 else f"{x:.2e}")
        df_chisq['test_statistic'] = df_chisq['test_statistic'].round(2)
        # Filter specific features to show significant/non-significant
        df_chisq_display = df_chisq[df_chisq['Feature'].isin(['job', 'education', 'contact', 'poutcome', 'housing', 'loan'])].copy()
        
        # Create Table Axis
        ax_tbl2 = fig.add_axes([0.09, 3.1, 6.7/8.5, 2.2/11])
        ax_tbl2.axis('off')
        
        tbl2_data = df_chisq_display[['Feature', 'test_statistic', 'p_value', 'significant']].values
        tbl2_headers = ['Categorical Feature', 'Chi-Square Stat', 'p-Value', 'Significant?']
        
        table2 = ax_tbl2.table(cellText=tbl2_data, colLabels=tbl2_headers, loc='center', cellLoc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.0, 1.4)
        for (row, col), cell in table2.get_celld().items():
            cell.set_edgecolor(c_border)
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(c_navy)
            else:
                # Color non-significant rows light red/orange for highlight
                feature_name = tbl2_data[row-1][0]
                if feature_name in ['housing', 'loan']:
                    cell.set_facecolor("#FFF0E6") # soft orange
                    if col == 3:
                        cell.set_text_props(weight='bold', color=c_accent_orange)
                else:
                    cell.set_facecolor(c_ice if row % 2 == 0 else 'white')
                    
    # Analysis Narrative at the bottom
    card_stat = Rectangle((0.8, 1.0), 6.9, 1.8, facecolor=c_ice, edgecolor=c_border, linewidth=1)
    ax.add_patch(card_stat)
    fig.text(1.0, 2.5, "INTERPRETATION FOR MANAGEMENT", color=c_navy, fontsize=11, fontweight="bold")
    
    stat_narrative = (
        "1. Numerical Significance: All 10 numerical features have a p-value of ~0.0, indicating clear behavioral "
        "separations. Age, previous calls, and interest rates differ significantly between subscriber cohorts.\n"
        "2. The Loan Paradox: In contrast to intuitive assumptions, housing and personal loan status are NOT "
        "statistically significant (p = 0.058 and p = 0.579). They do not independently affect the likelihood of "
        "subscribing. Call lists do not need filtering by property/personal debt profiles."
    )
    draw_text(fig, stat_narrative, 1.0, 2.2, width=64, fontsize=9, line_spacing=0.016)
    
    pdf_pages.savefig(fig)
    plt.close()
    
    # ---------------------------------------------------------
    # PAGE 4: DATA LEAKAGE CONTROL & PREPROCESSING PIPELINE
    # ---------------------------------------------------------
    print("Generating Page 4: Preprocessing and Target Leakage...")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    draw_header_banner(fig, ax, "PIPELINE ENGINEERING & LEAKAGE CONTROL", 4)
    
    # Narrative on Data Leakage
    leakage_box = Rectangle((0.8, 6.7), 6.9, 3.0, facecolor="#FFF0E6", edgecolor=c_accent_orange, linewidth=1.5)
    ax.add_patch(leakage_box)
    
    fig.text(1.0, 9.3, "CRITICAL COMPLIANCE: CONTROLLING TARGET LEAKAGE", color=c_accent_orange, fontsize=12, fontweight="bold")
    
    leakage_text = (
        "During exploratory analysis, the single strongest predictor of subscription was call 'duration' (length of the "
        "phone call in seconds). However, call duration is only determined *after* the telemarketer places a call and "
        "completes the conversation.\n\n"
        "Why duration was dropped from all model pipelines:\n"
        "* Production deployment constraint: To optimize a telemarketing lead list, the classifier must evaluate prospective "
        "customers *before* any calls are made. At prediction time, call duration is zero.\n"
        "* Model validation risk: Retaining duration yields artificially high metrics (91%+ accuracy, 65%+ F1) in testing "
        "but results in an completely useless model in production.\n"
        "* Solution: Dropping duration forces the model to rely only on demographics, contact history, and macroeconomic "
        "context, making it realistic and deployable."
    )
    
    paragraphs = leakage_text.split('\n\n')
    curr_y = 9.0
    for p in paragraphs:
        curr_y = draw_text(fig, p, 1.0, curr_y, width=64, fontsize=9.5, line_spacing=0.018)
        curr_y -= 0.04
        
    # Preprocessing Pipeline Steps
    fig.text(0.8, 6.0, "PIPELINE DESIGN & PREPROCESSING STEPS", color=c_navy, fontsize=12, fontweight="bold")
    
    pipeline_text = (
        "To ensure reproducible training and prevent feature distribution shift between train and test splits, we built "
        "a modular sklearn ColumnTransformer pipeline:"
    )
    draw_text(fig, pipeline_text, 0.8, 5.7, width=70, fontsize=9.5)
    
    # Visual Box for Pipeline Steps
    pipe_box = Rectangle((0.8, 1.2), 6.9, 4.1, facecolor=c_ice, edgecolor=c_border, linewidth=1)
    ax.add_patch(pipe_box)
    
    pipeline_steps_text = (
        "1. Train-Test Partitioning\n"
        "   - Data split into 80% training / 20% test subsets.\n"
        "   - Enabled target stratification to preserve the 11.27% positive class ratio in both splits.\n\n"
        "2. Numerical Transformations\n"
        "   - Scaled using StandardScaler to normalize distributions with different orders of magnitude.\n"
        "   - Applied to age, campaign, pdays, previous, and macroeconomic columns.\n\n"
        "3. Categorical Transformations\n"
        "   - Encoded using OneHotEncoder(handle_unknown='ignore') to dynamically handle categories unseen in training.\n"
        "   - Placeholders like 'unknown' are kept as unique levels to avoid data deletion.\n\n"
        "4. Handling Class Imbalance\n"
        "   - Tested Class-Weighted algorithms (penalizing minor class errors higher).\n"
        "   - Implemented SMOTE (Synthetic Minority Over-sampling Technique) in the pipeline to artificially balance "
        "the training set representation."
    )
    
    p_steps = pipeline_steps_text.split('\n\n')
    curr_y = 4.9
    for p in p_steps:
        curr_y = draw_text(fig, p, 1.0, curr_y, width=64, fontsize=9, line_spacing=0.016)
        curr_y -= 0.04
        
    pdf_pages.savefig(fig)
    plt.close()
    
    # ---------------------------------------------------------
    # PAGE 5: MODEL COMPARISONS LEADERBOARD
    # ---------------------------------------------------------
    print("Generating Page 5: Model Comparisons...")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    draw_header_banner(fig, ax, "MACHINE LEARNING LEADERBOARD & SELECTION", 5)
    
    # Load model results
    model_path = os.path.join(adv_dir, "advanced_model_results.csv")
    
    if os.path.exists(model_path):
        df_models = pd.read_csv(model_path)
        
        # Sort by F1-Score
        df_models_sorted = df_models.sort_values(by='F1_yes', ascending=False)
        
        # Plot model comparison chart (horizontal bar chart of top models F1)
        ax_models = fig.add_axes([0.1, 6.2, 6.5/8.5, 3.4/11])
        
        # Limit to key representative models for readable plot
        plot_models = df_models_sorted.head(8).copy()
        
        # Clean names for plotting
        plot_models['Short_Method'] = plot_models['Method'].apply(lambda x: x.replace(" Logistic Regression", " LogReg"))
        
        y_pos = np.arange(len(plot_models))
        ax_models.barh(y_pos, plot_models['F1_yes'] * 100, color=c_slate, height=0.4, label='F1-Score (Positive)')
        ax_models.barh(y_pos + 0.3, plot_models['Recall_yes'] * 100, color=c_accent_cyan, height=0.3, label='Recall (Positive)', alpha=0.7)
        
        ax_models.set_yticks(y_pos + 0.15)
        ax_models.set_yticklabels(plot_models['Short_Method'], fontsize=8, color=c_text_dark)
        ax_models.set_xlabel("Percentage (%)", fontsize=9, color=c_text_dark)
        ax_models.set_title("Key Model Performance Comparison (%)", color=c_navy, fontsize=11, fontweight="bold", pad=15)
        ax_models.spines['top'].set_visible(False)
        ax_models.spines['right'].set_visible(False)
        ax_models.spines['left'].set_color(c_border)
        ax_models.spines['bottom'].set_color(c_border)
        ax_models.tick_params(colors=c_text_dark, labelsize=8)
        ax_models.grid(axis='x', linestyle='--', alpha=0.5, color=c_border)
        ax_models.legend(loc='lower right', fontsize=8)
        
        # Display table of results below chart
        fig.text(0.8, 5.7, "MODEL LEADERBOARD (Sorted by F1-Score)", color=c_navy, fontsize=11, fontweight="bold")
        
        # Format table numbers
        df_table = df_models_sorted[['Method', 'Accuracy', 'Precision_yes', 'Recall_yes', 'F1_yes', 'ROC_AUC']].copy()
        for col in ['Accuracy', 'Precision_yes', 'Recall_yes', 'F1_yes', 'ROC_AUC']:
            df_table[col] = df_table[col].apply(lambda x: f"{x*100:.2f}%")
            
        ax_tbl3 = fig.add_axes([0.08, 1.0, 6.9/8.5, 4.3/11])
        ax_tbl3.axis('off')
        
        # Limit to top 8 rows to fit page perfectly
        table_data = df_table.head(8).values
        table_headers = ['Method', 'Acc.', 'Prec. (Yes)', 'Recall (Yes)', 'F1 (Yes)', 'ROC AUC']
        
        table3 = ax_tbl3.table(cellText=table_data, colLabels=table_headers, loc='center', cellLoc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(8)
        table3.scale(1.0, 1.3)
        
        # Column widths adjustment (make first column wider)
        table3.auto_set_column_width(col=list(range(len(table_headers))))
        
        for (row, col), cell in table3.get_celld().items():
            cell.set_edgecolor(c_border)
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(c_navy)
            else:
                method_name = table_data[row-1][0]
                if "Tuned Random Forest" in method_name:
                    cell.set_facecolor("#EBFBFA") # very soft teal
                    cell.set_text_props(weight='bold', color=c_navy)
                else:
                    cell.set_facecolor(c_ice if row % 2 == 0 else 'white')
                    
    pdf_pages.savefig(fig)
    plt.close()
    
    # ---------------------------------------------------------
    # PAGE 6: FEATURE IMPORTANCES & BUSINESS RECOMMENDATIONS
    # ---------------------------------------------------------
    print("Generating Page 6: Feature Importances & Business Actions...")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    draw_header_banner(fig, ax, "BUSINESS INSIGHTS & ACTIONABLE RECOMMENDATIONS", 6)
    
    # Load and Plot Feature Importances
    feat_path = os.path.join(model_dir, "feature_importance.csv")
    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path)
        # Select top 10 features
        df_feat_top = df_feat.head(10).sort_values(by='Importance', ascending=True)
        
        ax_feat = fig.add_axes([0.1, 6.2, 6.5/8.5, 3.4/11])
        ax_feat.barh(df_feat_top['Feature'], df_feat_top['Importance'] * 100, color=c_slate, height=0.5)
        ax_feat.set_title("Random Forest Top 10 Feature Importances (%)", color=c_navy, fontsize=11, fontweight="bold", pad=15)
        ax_feat.set_xlabel("Importance Percentage (%)", fontsize=9, color=c_text_dark)
        ax_feat.spines['top'].set_visible(False)
        ax_feat.spines['right'].set_visible(False)
        ax_feat.spines['left'].set_color(c_border)
        ax_feat.spines['bottom'].set_color(c_border)
        ax_feat.tick_params(colors=c_text_dark, labelsize=8)
        ax_feat.grid(axis='x', linestyle='--', alpha=0.5, color=c_border)
        
    # Recommendations Card
    rec_card = Rectangle((0.8, 1.0), 6.9, 4.6, facecolor=c_ice, edgecolor=c_border, linewidth=1)
    ax.add_patch(rec_card)
    
    fig.text(1.0, 5.2, "STRATEGIC CAMPAIGN ACTION PLAN", color=c_navy, fontsize=12, fontweight="bold")
    
    recs_text = (
        "1. Deploy the Tuned Random Forest Classifier:\n"
        "   - Use the model's scores to prioritize leads. Instead of calling the entire database, target only clients "
        "with positive classification.\n"
        "   - This cuts database contacts by over 80%, lowering cost-per-contact and maximizing agent efficiency.\n\n"
        "2. Capitalize on Economic Windows:\n"
        "   - Macroeconomic variables (euribor3m, nr.employed, emp.var.rate) represent over 46% of model predictive power.\n"
        "   - Clients are statistically far more likely to subscribe when interest rates are low. Direct marketing campaigns "
        "should increase volume when interest rate contractions occur.\n\n"
        "3. Adjust Operational Call Thresholds:\n"
        "   - At standard 0.5 threshold, model yields 44.22% precision and 60.99% recall.\n"
        "   - If campaign budget is larger, shift threshold to 0.4. This boosts subscriber capture to 73.38% (Recall) "
        "while keeping call efficiency significantly above random calling.\n\n"
        "4. Enforce Campaign Call Capping:\n"
        "   - The statistical analysis showed call fatigue is high. Capping contacts per client to a maximum of 3-4 calls "
        "per campaign maintains customer trust and reduces marketing waste."
    )
    
    paragraphs = recs_text.split('\n\n')
    curr_y = 4.9
    for p in paragraphs:
        curr_y = draw_text(fig, p, 1.0, curr_y, width=64, fontsize=9.5, line_spacing=0.016)
        curr_y -= 0.05
        
    pdf_pages.savefig(fig)
    plt.close()
    
    # Close PDF Writer
    pdf_pages.close()
    print("PDF Generation complete. Output saved to:", output_path)

if __name__ == "__main__":
    # Absolute paths relative to the current script execution directory
    workspace_dir = r"c:\Users\DELL\OneDrive\Desktop\TenX"
    eda_dir = os.path.join(workspace_dir, "eda_outputs")
    model_dir = os.path.join(workspace_dir, "model_outputs")
    adv_dir = os.path.join(workspace_dir, "advanced_outputs")
    output_pdf = os.path.join(workspace_dir, "Bank_Marketing_Project_Report.pdf")
    
    create_pdf_report(output_pdf, eda_dir, model_dir, adv_dir)

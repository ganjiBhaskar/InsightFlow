import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import streamlit as st
from utils import get_llm_insight

def summarize_data(df):
    """Generate summary statistics including basic and detailed stats."""
    basic_stats = df.describe()
    detailed_stats = df.describe(include='all')
    missing_values = df.isnull().sum()
    
    # Generate two insights for descriptive statistics
    insights = [
        f"The mean of numeric columns ranges from {basic_stats.loc['mean'].min():.2f} to {basic_stats.loc['mean'].max():.2f}.",
        f"The column with the most missing values is '{missing_values.idxmax()}' with {missing_values.max()} missing entries."
    ]
    return basic_stats, detailed_stats, missing_values, insights

def initial_exploratory_plots(df, selected_numeric, selected_categorical):
    """Generate initial exploratory plots like histograms, boxplots, and count plots."""
    st.subheader("Initial Exploratory Plots")
    insights = []
    
    # Histograms for Selected Numeric Columns
    if len(selected_numeric) > 0:
        st.write("Histograms for Selected Numeric Columns:")
        for col in selected_numeric:
            if col in df.columns:  # Ensure the column exists in the DataFrame
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[col], kde=True, bins=30, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)
                
                # LLM Analysis for Histogram
                prompt = (
                    f"Analyze the histogram of '{col}'. "
                    f"The distribution appears to be {'normal' if df[col].skew() < 0.5 else 'skewed'}. "
                    "Provide insights in natural language."
                )
                llm_insight = get_llm_insight(prompt)
                st.write(f"LLM Insight: {llm_insight}")
                insights.append(llm_insight)
            else:
                st.warning(f"Column '{col}' not found in the dataset.")
    
    # Boxplots for Selected Numeric Columns
    if len(selected_numeric) > 0:
        st.write("Boxplots for Selected Numeric Columns:")
        for col in selected_numeric:
            if col in df.columns:  # Ensure the column exists in the DataFrame
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)
                
                # LLM Analysis for Boxplot
                prompt = (
                    f"Analyze the boxplot of '{col}'. "
                    f"The interquartile range is from {df[col].quantile(0.25)} to {df[col].quantile(0.75)}. "
                    "Provide insights in natural language."
                )
                llm_insight = get_llm_insight(prompt)
                st.write(f"LLM Insight: {llm_insight}")
                insights.append(llm_insight)
            else:
                st.warning(f"Column '{col}' not found in the dataset.")
    
    # Count Plots for Selected Categorical Columns
    if len(selected_categorical) > 0:
        st.write("Count Plots for Selected Categorical Columns:")
        for col in selected_categorical:
            if col in df.columns:  # Ensure the column exists in the DataFrame
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(data=df, x=col, palette='coolwarm', ax=ax)
                plt.xticks(rotation=45)
                ax.set_title(f"Count Plot of {col}")
                st.pyplot(fig)
                
                # LLM Analysis for Count Plot
                prompt = (
                    f"Analyze the count plot of '{col}'. "
                    f"The most frequent category is '{df[col].value_counts().idxmax()}' with {df[col].value_counts().max()} occurrences. "
                    "Provide insights in natural language."
                )
                llm_insight = get_llm_insight(prompt)
                st.write(f"LLM Insight: {llm_insight}")
                insights.append(llm_insight)
            else:
                st.warning(f"Column '{col}' not found in the dataset.")
    
    return insights

def generate_combinations(df, target_column, use_binning=False):
    """Generate crosstab, pivot table, and groupby summaries based on data type."""
    insights = []
    
    # Check for categorical and numeric columns
    object_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Crosstab Analysis (only if there are at least 2 categorical columns)
    if len(object_cols) > 1 and target_column in object_cols:
        st.subheader("Crosstab Analysis")
        for i in object_cols[1:-1]:  # Exclude the first and last categorical columns
            st.write(f"{target_column} with respect to the {i}")
            crosstab_result = pd.crosstab(df[target_column], df[i], margins=True)
            st.write(crosstab_result)
            
            # LLM Analysis for Crosstab
            prompt = (
                f"Analyze the crosstab between '{target_column}' and '{i}'. "
                f"The crosstab shows the following counts:\n{crosstab_result.to_string()}. "
                "Provide insights in natural language."
            )
            llm_insight = get_llm_insight(prompt)
            st.write(f"LLM Insight: {llm_insight}")
            insights.append(llm_insight)
            st.write("*" * 100)
    else:
        st.info("Skipping crosstab analysis due to insufficient categorical columns.")
    
    # Pivot Table Analysis (only if there are at least 2 categorical columns and numeric columns)
    if len(object_cols) > 0 and len(numeric_cols) > 0:
        st.subheader("Pivot Table Analysis")
        pivot_result = df.pivot_table(index=object_cols[0], values=numeric_cols[0], aggfunc='mean')
        st.write(pivot_result)
        
        # LLM Analysis for Pivot Table
        prompt = (
            f"Analyze the pivot table where '{object_cols[0]}' is grouped and '{numeric_cols[0]}' is aggregated by mean. "
            f"The pivot table shows the following averages:\n{pivot_result.to_string()}. "
            "Provide insights in natural language."
        )
        llm_insight = get_llm_insight(prompt)
        st.write(f"LLM Insight: {llm_insight}")
        insights.append(llm_insight)
        st.write("*" * 100)
    else:
        st.info("Skipping pivot table analysis due to insufficient categorical or numeric columns.")
    
    # Groupby Analysis (group by object columns and aggregate numeric columns)
    if len(object_cols) > 0 and len(numeric_cols) > 0:
        st.subheader("Groupby Analysis")
        for obj_col in object_cols:
            for num_col in numeric_cols:
                st.write(f"Groupby analysis of '{obj_col}' vs '{num_col}'")
                groupby_result = df.groupby(obj_col)[num_col].mean().reset_index()
                st.write(groupby_result)
                
                # Bar Plot for Groupby Results
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=groupby_result, x=obj_col, y=num_col, palette='coolwarm', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # LLM Analysis for Groupby
                prompt = (
                    f"Analyze the bar plot showing the average '{num_col}' for each category in '{obj_col}'. "
                    f"The highest average value is {groupby_result[num_col].max():.2f} for the category '{groupby_result.iloc[groupby_result[num_col].idxmax()][obj_col]}'. "
                    "Provide insights in natural language."
                )
                llm_insight = get_llm_insight(prompt)
                st.write(f"LLM Insight: {llm_insight}")
                insights.append(llm_insight)
                st.write("*" * 100)
    
    return insights

def statistical_analysis(df, target_column, analysis_type):
    """Perform statistical analysis based on user choice (regression or classification)."""
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if analysis_type == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            st.error(f"Target column '{target_column}' is not numeric. Regression requires a numeric target column.")
            return None, []
        
        # Perform OLS regression
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        
        # Generate insights
        insights = [
            f"The R-squared value of the regression model is {model.rsquared:.2f}, indicating the proportion of variance explained.",
            f"The most significant predictor is '{model.pvalues.idxmin()}' with a p-value of {model.pvalues.min():.4f}."
        ]
        return model.summary(), insights
    
    elif analysis_type == "classification":
        if not pd.api.types.is_object_dtype(y) and not pd.api.types.is_categorical_dtype(y):
            st.error(f"Target column '{target_column}' is not categorical. Classification requires a categorical target column.")
            return None, []
        
        # Perform logistic regression
        X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Generate classification report
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Generate insights
        insights = [
            f"The accuracy of the classification model is {report['accuracy']:.2f}.",
            f"The most important feature is '{X.columns[model.coef_[0].argmax()]}' with a coefficient of {model.coef_[0].max():.4f}."
        ]
        return report_df, insights

def visualize_data(df):
    """Create visualizations and generate insights."""
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # LLM Analysis for Heatmap
    correlation_matrix = df.corr(numeric_only=True)
    high_corr_pairs = [
        (i, j, correlation_matrix.loc[i, j])
        for i in correlation_matrix.columns
        for j in correlation_matrix.columns
        if i != j and abs(correlation_matrix.loc[i, j]) > 0.7
    ]
    prompt = (
        "Analyze the heatmap of feature correlations. "
        f"The following pairs of features have strong correlations:\n{high_corr_pairs}. "
        "Provide insights in natural language."
    )
    llm_insight = get_llm_insight(prompt)
    st.write(f"LLM Insight: {llm_insight}")
    
    st.subheader("Pairplot of Features")
    pairplot_fig = sns.pairplot(df)
    st.pyplot(pairplot_fig)

def final_summary(df, insights):
    """Generate a final summary using the LLM."""
    st.subheader("Final Comprehensive Summary")
    prompt = (
        "Summarize the key findings from the dataset analysis. "
        "The dataset contains the following columns:\n"
        f"{df.columns.tolist()}\n"
        "The following insights were generated during the analysis:\n"
        f"{insights}\n"
        "Provide a cohesive summary in natural language."
    )
    llm_insight = get_llm_insight(prompt)
    st.write(llm_insight)
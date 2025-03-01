import streamlit as st
from app.data_loader import load_data
from app.data_cleaning import clean_data
from app.data_analysis import (
    summarize_data,
    initial_exploratory_plots,
    generate_combinations,
    statistical_analysis,
    visualize_data,
    final_summary,
)
from utils.utils import generate_pdf_report

def main():
    """Streamlit App Execution."""
    st.title("Automated Data Analysis Agent")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            if isinstance(df, pd.DataFrame):
                st.subheader("Data Preview")
                st.write(df.head())
                
                # Summarize Data
                basic_stats, detailed_stats, missing_values, stats_insights = summarize_data(df)
                st.subheader("Basic Statistics")
                st.write(basic_stats)
                st.subheader("Statistics Insights")
                for insight in stats_insights[:2]:  # Display only two insights
                    st.write(insight)
                
                st.subheader("Detailed Statistics")
                st.write(detailed_stats)
                
                st.subheader("Missing Values")
                st.write(missing_values)
                
                # Clean Data
                cleaned_df = clean_data(df)
                
                # Target Column and Analysis Type Selection
                target_column = st.sidebar.selectbox("Select the target column (predicted column):", cleaned_df.columns)
                analysis_type = st.sidebar.selectbox("Select analysis type:", ["regression", "classification"])
                
                # Column Selection for Visualizations
                numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
                object_cols = cleaned_df.select_dtypes(include=['object']).columns
                
                selected_numeric = st.multiselect("Select numeric columns for histograms and boxplots:", numeric_cols)
                selected_categorical = st.multiselect("Select categorical columns for count plots:", object_cols)
                
                # Ask whether to build a model
                build_model = st.sidebar.checkbox("Build a predictive model?", value=False)
                
                # Checkbox for PDF Export
                export_to_pdf = st.sidebar.checkbox("Export analysis to PDF?", value=False)
                
                # Generate Analysis Button
                if st.sidebar.button("Generate Analysis"):
                    st.success("Starting analysis...")
                    
                    # Initial Exploratory Plots
                    initial_insights = initial_exploratory_plots(cleaned_df, selected_numeric, selected_categorical)
                    
                    # Checkbox for enabling binning
                    use_binning = st.sidebar.checkbox("Enable binning for numerical data (quantiles)", value=False)
                    
                    st.subheader("Crosstab, Pivot Table, and Groupby Analysis")
                    combination_insights = generate_combinations(cleaned_df, target_column=target_column, use_binning=use_binning)
                    
                    visualize_data(cleaned_df)
                    
                    # Statistical Analysis (only if the user chooses to build a model)
                    stats_summary = None
                    if build_model:
                        stats_summary, stats_insights = statistical_analysis(cleaned_df, target_column=target_column, analysis_type=analysis_type)
                        if stats_summary is not None:
                            st.subheader(f"{analysis_type.capitalize()} Results")
                            st.write(stats_summary)
                            st.subheader(f"{analysis_type.capitalize()} Insights")
                            for insight in stats_insights[:2]:  # Display only two insights
                                st.write(insight)
                        else:
                            st.warning("Statistical analysis could not be performed due to incompatible target column.")
                    else:
                        st.info("Skipping model building as per user choice.")
                        stats_insights = []
                    
                    # Combine all insights
                    all_insights = stats_insights + combination_insights + initial_insights
                    
                    # Final Comprehensive Summary
                    final_summary(cleaned_df, all_insights)
                    
                    st.success("Analysis complete!")
                    
                    # Export to PDF if the checkbox is selected
                    if export_to_pdf:
                        # Collect plots for inclusion in the PDF
                        plots = []
                        
                        # Example: Add histograms for numeric columns
                        for col in selected_numeric:
                            if col in cleaned_df.columns:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                sns.histplot(cleaned_df[col], kde=True, bins=30, ax=ax)
                                ax.set_title(f"Histogram of {col}")
                                plots.append(fig)
                        
                        # Example: Add count plots for categorical columns
                        for col in selected_categorical:
                            if col in cleaned_df.columns:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                sns.countplot(data=cleaned_df, x=col, palette='coolwarm', ax=ax)
                                plt.xticks(rotation=45)
                                ax.set_title(f"Count Plot of {col}")
                                plots.append(fig)
                        
                        # Example: Add correlation heatmap
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.heatmap(cleaned_df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
                        ax.set_title("Feature Correlation Heatmap")
                        plots.append(fig)
                        
                        # Generate the PDF report with plots
                        pdf_path = generate_pdf_report(cleaned_df, all_insights, analysis_type, stats_summary, plots)
                        
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=f,
                                    file_name="data_analysis_report.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            st.error("Failed to generate the PDF report. Please check the logs.")
            else:
                st.error("Could not process the dataset. Converting to CSV format.")
                df.to_csv("converted_dataset.csv", index=False)
                st.success("Dataset converted and saved as converted_dataset.csv")
    
    st.success("Ready for analysis!")

if __name__ == "__main__":
    main()
import pandas as pd
import streamlit as st

def clean_data(df):
    """Clean the dataset based on user input from the sidebar."""
    st.sidebar.subheader("Data Cleaning Options")
    cleaned_df = df.copy()
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Display missing values
    st.sidebar.write("Columns with Missing Values:")
    st.sidebar.write(missing_percentage[missing_percentage > 0])
    
    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            st.sidebar.write(f"Column '{col}' has {df[col].isnull().sum()} missing values ({missing_percentage[col]:.2f}%).")
            method = st.sidebar.selectbox(
                f"How would you like to handle missing values in '{col}'?",
                ["Do Nothing", "Fill with Mean", "Fill with Median", "Drop Rows", "Drop Column"],
                key=f"numeric_{col}"
            )
            if method == "Fill with Mean":
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif method == "Fill with Median":
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif method == "Drop Rows":
                cleaned_df.dropna(subset=[col], inplace=True)
            elif method == "Drop Column":
                cleaned_df.drop(columns=[col], inplace=True)
    
    # Handle missing values for categorical columns
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].isnull().sum() > 0:
            st.sidebar.write(f"Column '{col}' has {df[col].isnull().sum()} missing values ({missing_percentage[col]:.2f}%).")
            method = st.sidebar.selectbox(
                f"How would you like to handle missing values in '{col}'?",
                ["Do Nothing", "Fill with Mode", "Drop Rows", "Drop Column"],
                key=f"categorical_{col}"
            )
            if method == "Fill with Mode":
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            elif method == "Drop Rows":
                cleaned_df.dropna(subset=[col], inplace=True)
            elif method == "Drop Column":
                cleaned_df.drop(columns=[col], inplace=True)
    
    st.sidebar.success("Data cleaning options set!")
    
    # Save cleaned data to CSV if changes were made
    if not cleaned_df.equals(df):  # Check if the DataFrame was modified
        cleaned_df.to_csv("cleaned_dataset.csv", index=False)
        st.sidebar.success("Cleaned dataset saved as 'cleaned_dataset.csv'")
    
    return cleaned_df
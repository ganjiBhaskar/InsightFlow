import pandas as pd
import streamlit as st

def load_data(file):
    """Load dataset from different formats with flexible delimiter and encoding."""
    encodings = ["utf-8", "latin1", "ISO-8859-1", "utf-16", "utf-32", "cp1252"]
    try:
        if file.name.endswith('.csv'):
            for encoding in encodings:
                try:
                    return pd.read_csv(file, delimiter=None, encoding=encoding, engine='python')
                except Exception:
                    continue
        elif file.name.endswith('.xlsx'):
            for encoding in encodings:
                try:
                    return pd.read_excel(file, engine='openpyxl')
                except Exception:
                    continue
        elif file.name.endswith('.json'):
            for encoding in encodings:
                try:
                    return pd.read_json(file, encoding=encoding)
                except Exception:
                    continue
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
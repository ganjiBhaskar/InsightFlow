import os
from fpdf import FPDF
import matplotlib.pyplot as plt
from langchain_huggingface import HuggingFaceEndpoint

# Load Hugging Face LLM for text analysis
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=512, temperature=0.7, token=hf_token)

def get_llm_insight(prompt):
    """Centralized function to get LLM insights."""
    llm_response = llm(prompt)
    return llm_response[0]['generated_text'] if isinstance(llm_response, list) else llm_response

def save_plot_as_image(fig, filename):
    """Save a Matplotlib figure as an image file."""
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)

def generate_pdf_report(df, insights, analysis_type, stats_summary=None, plots=None):
    """
    Generate a PDF report containing the analysis results.
    
    Args:
        df (pd.DataFrame): The dataset being analyzed.
        insights (list): List of insights generated during the analysis.
        analysis_type (str): Type of analysis performed ("regression" or "classification").
        stats_summary (pd.DataFrame, optional): Statistical summary of the analysis.
        plots (list, optional): List of Matplotlib figures to include in the PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Automated Data Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Section: Dataset Overview
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="1. Dataset Overview", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Columns: {', '.join(df.columns)}\nRows: {len(df)}\n")
    pdf.ln(5)

    # Section: Basic Statistics
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="2. Basic Statistics", ln=True)
    pdf.set_font("Arial", size=12)
    basic_stats = df.describe().to_string()
    pdf.multi_cell(0, 10, txt=basic_stats)
    pdf.ln(5)

    # Section: Missing Values
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="3. Missing Values", ln=True)
    pdf.set_font("Arial", size=12)
    missing_values = df.isnull().sum()
    pdf.multi_cell(0, 10, txt=missing_values.to_string())
    pdf.ln(5)

    # Section: Key Insights
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="4. Key Insights", ln=True)
    pdf.set_font("Arial", size=12)
    for i, insight in enumerate(insights, 1):
        pdf.multi_cell(0, 10, txt=f"{i}. {insight}")
    pdf.ln(5)

    # Section: Statistical Analysis Results
    if stats_summary is not None:
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt=f"5. {analysis_type.capitalize()} Results", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=stats_summary.to_string())
        pdf.ln(5)

    # Section: Visualizations
    if plots is not None:
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt="6. Visualizations", ln=True)
        pdf.set_font("Arial", size=12)
        
        for i, fig in enumerate(plots, 1):
            # Save the plot as an image
            plot_filename = f"plot_{i}.png"
            save_plot_as_image(fig, plot_filename)
            
            # Add the image to the PDF
            pdf.cell(200, 10, txt=f"Visualization {i}", ln=True)
            pdf.image(plot_filename, x=10, y=None, w=180)
            pdf.ln(5)
    
    # Save the PDF
    pdf_output_path = "data_analysis_report.pdf"
    pdf.output(pdf_output_path)
    
    # Clean up temporary image files
    if plots is not None:
        for i in range(1, len(plots) + 1):
            os.remove(f"plot_{i}.png")
    
    return pdf_output_path
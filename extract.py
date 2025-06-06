# app.py
import os
import re
import json
import pdfplumber
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import openai
from openai import OpenAI
from typing import Dict, List, Any

# ─── Configuration ─────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

# ─── Utility Functions ─────────────────────────
def clean_numeric_value(value: str) -> float:
    """Extract numeric value from string, handling various formats"""
    if not value or value == "" or value == "0":
        return 0.0
    
    # Remove currency symbols, commas, and spaces
    cleaned = re.sub(r'[€$,\s]', '', str(value))
    
    # Handle negative values in parentheses
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    # Extract numeric value
    numeric_match = re.search(r'-?\d+\.?\d*', cleaned)
    if numeric_match:
        return float(numeric_match.group())
    return 0.0

def calculate_netherlands_tax(taxable_income: float) -> float:
    """Calculate Netherlands corporate tax based on 2024 rates"""
    # Tax is only calculated on positive taxable income
    if taxable_income <= 0:
        return 0.0
    elif taxable_income <= 200000:
        return taxable_income * 0.19
    else:
        return (200000 * 0.19) + ((taxable_income - 200000) * 0.258)

def extract_financial_data_with_ai(text: str, tables_data: str) -> Dict[str, Any]:
    """Enhanced AI extraction with better prompting and validation"""
    
    ENHANCED_SYSTEM_PROMPT = """
    You are an expert Corporate Tax Analyzer AI. Extract financial data accurately from the provided document.

    EXTRACTION RULES:
    1. Company Name: Look for company headers, "B.V.", "Ltd", "Inc", "Corp"
    2. Country: Find country from addresses or legal entity information
    3. Total Revenue: Find "Revenue", "Sales", "Turnover", "Income", "Omzet" (can be 0 if not found)
    4. Total Expenses: Look for "Expenses", "Costs", "Uitgaven", "Kosten" (expenses are ALWAYS positive numbers)
    5. Depreciation: Find "Depreciation", "Afschrijving", "Amortization"
    6. Deductions: Look for "Deductions", "Aftrekposten", "Tax Deductions"

    CRITICAL RULES:
    - ALL financial amounts should be positive numbers (even expenses)
    - If revenue is 0 or very low, and expenses exist, this is normal for new companies
    - Look carefully in tables for line items like "Kosten", "Uitgaven", "Expenses"
    - Expenses include: office costs, salaries, rent, utilities, professional fees, etc.
    - Revenue might be 0 for startup companies or holding companies
    - Return "0" if field not found, never leave empty

    CALCULATION:
    net_taxable_income = total_revenue - total_expenses - depreciation - deductions
    (This CAN be negative for loss-making companies)

    Return ONLY valid JSON:
    {
      "company_name": "string",
      "country": "string", 
      "total_revenue": "string_number",
      "total_expenses": "string_number",
      "depreciation": "string_number",
      "deductions": "string_number",
      "net_taxable_income": "calculated_string_number",
      "final_tax_owed": "string_number"
    }
    """

    combined_data = f"DOCUMENT TEXT:\n{text}\n\nTABLE DATA:\n{tables_data}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ENHANCED_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": combined_data[:15000]}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean JSON response
        if result.startswith("```"):
            result = re.sub(r'```json\n?|```\n?', '', result).strip()
        
        # Parse and validate JSON
        parsed_data = json.loads(result)
        
        # Validate and recalculate to ensure accuracy
        revenue = clean_numeric_value(parsed_data.get("total_revenue", "0"))
        expenses = clean_numeric_value(parsed_data.get("total_expenses", "0"))
        depreciation = clean_numeric_value(parsed_data.get("depreciation", "0"))
        deductions = clean_numeric_value(parsed_data.get("deductions", "0"))
        
        # Calculate taxable income (can be negative)
        net_taxable = revenue - expenses - depreciation - deductions
        final_tax = calculate_netherlands_tax(net_taxable)
        
        # Update with recalculated values
        parsed_data["total_revenue"] = str(int(round(revenue)))
        parsed_data["total_expenses"] = str(int(round(expenses)))
        parsed_data["depreciation"] = str(int(round(depreciation)))
        parsed_data["deductions"] = str(int(round(deductions)))
        parsed_data["net_taxable_income"] = str(int(round(net_taxable)))
        parsed_data["final_tax_owed"] = str(int(round(final_tax)))
        
        return parsed_data
        
    except Exception as e:
        st.error(f"AI Processing Error: {str(e)}")
        return {
            "company_name": "",
            "country": "",
            "total_revenue": "0",
            "total_expenses": "0", 
            "depreciation": "0",
            "deductions": "0",
            "net_taxable_income": "0",
            "final_tax_owed": "0"
        }

def format_currency(amount: str) -> str:
    """Format number as currency string with proper negative handling"""
    try:
        num = float(amount)
        if num < 0:
            return f"-€{abs(num):,.0f}"
        return f"€{num:,.0f}"
    except:
        return "€0"

def create_results_table(financial_data: Dict[str, Any]) -> pd.DataFrame:
    """Create a clean results table"""
    
    # Create the table data
    table_data = [
        ["Company Name", financial_data.get("company_name", "Not Found")],
        ["Country", financial_data.get("country", "Not Found")],
        ["Total Revenue", format_currency(financial_data.get("total_revenue", "0"))],
        ["Total Expenses", format_currency(financial_data.get("total_expenses", "0"))],
        ["Depreciation", format_currency(financial_data.get("depreciation", "0"))],
        ["Deductions", format_currency(financial_data.get("deductions", "0"))],
        ["Net Taxable Income", format_currency(financial_data.get("net_taxable_income", "0"))],
        ["Final Tax Owed", format_currency(financial_data.get("final_tax_owed", "0"))]
    ]
    
    df = pd.DataFrame(table_data, columns=["Field", "Value"])
    return df

# ─── Main Streamlit App ─────────────────────────
def main():
    st.set_page_config(
        page_title="Corporate Tax Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Corporate Tax Analyzer")
    st.markdown("**Upload your financial document and get instant tax analysis**")
    st.markdown("---")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "📁 Upload Financial Statement PDF", 
        type=["pdf"],
        help="Upload PDF containing financial statements, trial balances, or tax documents"
    )
    
    if uploaded_file:
        st.success("✅ PDF uploaded successfully!")
        
        # Automatically process the PDF when uploaded
        with st.spinner("🔄 Processing document..."):
            # Extract content
            full_text = ""
            all_tables = []
            tables_text = ""
            
            with pdfplumber.open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        if table:
                            df = pd.DataFrame(table)
                            all_tables.append(df)
                            tables_text += f"\nTable {j+1} (Page {i+1}):\n{df.to_string()}\n"
            
            # AI Analysis
            financial_data = extract_financial_data_with_ai(full_text, tables_text)
        
        st.success("✅ Analysis Complete!")
        
        # Display results as table
        st.subheader("📈 Financial Analysis Results")
        
        # Create and display the results table
        results_df = create_results_table(financial_data)
        
        # Style the table
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Field": st.column_config.TextColumn("Field", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium")
            }
        )
        
        # Show calculation explanation if there's a negative taxable income
        net_taxable = float(financial_data.get("net_taxable_income", "0"))
        if net_taxable < 0:
            st.info(f"""
            ℹ️ **Negative Taxable Income Explanation:**
            This company has a net loss of {format_currency(str(abs(net_taxable)))} for the period.
            
            **Calculation:** €{financial_data.get('total_revenue', '0')} (Revenue) - €{financial_data.get('total_expenses', '0')} (Expenses) - €{financial_data.get('depreciation', '0')} (Depreciation) - €{financial_data.get('deductions', '0')} (Deductions) = {format_currency(financial_data.get('net_taxable_income', '0'))}
            
            **Tax Implication:** No corporate tax is owed on losses. The loss can typically be carried forward to offset future profits.
            """)
        
        # Export functionality
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Create export data
            export_data = {
                "Analysis Results": financial_data,
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.download_button(
                label="📋 Download Results",
                data=json.dumps(export_data, indent=2),
                file_name=f"tax_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col1:
            st.markdown("**Netherlands Corporate Tax Rates:** First €200,000 at 19%, remainder at 25.8%")

if __name__ == "__main__":
    main()
import os
import re
import json
import pdfplumber
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# ─── Utility Functions ─────────────────────────
def clean_numeric_value(value: str) -> float:
    if not value or value == "" or value == "0":
        return 0.0
    cleaned = re.sub(r'[\u20ac$,%\s]', '', str(value))
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    match = re.search(r'-?\d+\.?\d*', cleaned)
    return float(match.group()) if match else 0.0

def calculate_netherlands_tax(taxable_income: float) -> float:
    if taxable_income <= 0:
        return 0.0
    if taxable_income <= 200000:
        return taxable_income * 0.19
    return 200000 * 0.19 + (taxable_income - 200000) * 0.258

def extract_financial_data_with_ai(text: str, tables_data: str) -> Dict[str, Any]:
    PROMPT = """
You are a Corporate Tax Analyzer AI assistant.
Your job is to extract clean, structured financial data from Dutch corporate tax documents. These documents may include trial balances, profit-loss statements, or invoices in either PDF or CSV form.

From the given document text, extract and return only the following fields in valid JSON format:
{
  "company_name": "",
  "country": "",
  "total_revenue": "",
  "total_expenses": "",
  "depreciation": "",
  "deductions": "",
  "net_taxable_income": "",
  "final_tax_owed": ""
}

Instructions:
- 'country' refers to the **client's country** (recipient), if mentioned in the document.
- 'net_taxable_income' = total_revenue - total_expenses - depreciation - deductions
- Use Netherlands tax rules:
    - ≤ 200k €: 19%
    - > 200k €: 25.8%
- Return only valid JSON with double quotes.
"""
    combined = f"DOCUMENT TEXT:\n{text}\n\nTABLE DATA:\n{tables_data}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": PROMPT.strip()},
                {"role": "user", "content": combined[:15000]}
            ],
            temperature=0,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        if result.startswith("```"):
            result = re.sub(r'```json\n?|```', '', result).strip()
        data = json.loads(result)

        revenue = clean_numeric_value(data.get("total_revenue", "0"))
        expenses = clean_numeric_value(data.get("total_expenses", "0"))
        depreciation = clean_numeric_value(data.get("depreciation", "0"))
        deductions = clean_numeric_value(data.get("deductions", "0"))
        net_income = revenue - expenses - depreciation - deductions
        tax = calculate_netherlands_tax(net_income)

        return {
            "company_name": data.get("company_name", ""),
            "country": data.get("country", ""),
            "total_revenue": str(int(revenue)),
            "total_expenses": str(int(expenses)),
            "depreciation": str(int(depreciation)),
            "deductions": str(int(deductions)),
            "net_taxable_income": str(int(net_income)),
            "final_tax_owed": str(int(tax))
        }
    except Exception as e:
        st.error(f"AI Processing Error: {e}")
        return {k: "" if k in ["company_name", "country"] else "0" for k in [
            "company_name", "country", "total_revenue", "total_expenses", "depreciation",
            "deductions", "net_taxable_income", "final_tax_owed"]}

def format_currency(amount: str) -> str:
    try:
        num = float(amount)
        return f"-\u20ac{abs(num):,.0f}" if num < 0 else f"\u20ac{num:,.0f}"
    except:
        return "\u20ac0"

def create_results_table(financial_data: Dict[str, Any]) -> pd.DataFrame:
    rows = [[k.replace("_", " ").title(), format_currency(financial_data[k])] for k in [
        "total_revenue", "total_expenses", "depreciation", "deductions",
        "net_taxable_income", "final_tax_owed"]]
    rows.insert(0, ["Company Name", financial_data.get("company_name", "")])
    rows.insert(1, ["Country", financial_data.get("country", "")])
    return pd.DataFrame(rows, columns=["Field", "Value"])

# ─── Main App ───────────────────────────────────
def main():
    st.set_page_config(page_title="Corporate Tax Analyzer", layout="wide")
    st.title("📊 Corporate Tax Analyzer")
    st.markdown("**Upload a corporate financial document (PDF/CSV)**")

    file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
    if not file:
        return

    full_text, tables_text = "", ""

    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                tables = page.extract_tables()
                for t in tables:
                    df = pd.DataFrame(t)
                    tables_text += df.to_string(index=False) + "\n"
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        full_text = df.to_string(index=False)
        tables_text = full_text

    with st.spinner("Processing with GPT-4..."):
        result = extract_financial_data_with_ai(full_text, tables_text)

    st.subheader("📈 Financial Analysis Results")
    df_result = create_results_table(result)
    st.dataframe(df_result, use_container_width=True)

    net = float(result.get("net_taxable_income", 0))
    if net < 0:
        st.warning("The company incurred a net loss. No tax is owed this year.")

    st.download_button("📥 Download JSON Report", json.dumps(result, indent=2),
                       file_name="tax_analysis.json", mime="application/json")

if __name__ == "__main__":
    main()

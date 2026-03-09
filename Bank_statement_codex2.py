import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import PyPDF2


def capitec_pdf_to_clean_text(uploaded_file):

    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    # ----- Capitec specific cleanup -----

    num_start = text.find("Date Description Category")
    new_text = text[num_start:]

    pattern = r"\* Includes VAT.*?Fee\*Balance"
    new2text = re.sub(pattern, "", new_text, flags=re.DOTALL)

    lines = new2text.split("\n")
    cleaned_lines = []

    date_pattern = r"\d{2}/\d{2}/\d{4}"

    for line in lines:

        if re.match(date_pattern, line):
            cleaned_lines.append(line)

        else:

            if line.startswith("Date Description Category Money In Money Out Fee*Balance"):
                cleaned_lines.append(line)

            elif cleaned_lines:
                cleaned_lines[-1] += " " + line.strip()

    new_text_no_cont = "\n".join(cleaned_lines)

    final_text = re.sub(
        r"\s*\* Includes VAT at \d+%?\*",
        "",
        new_text_no_cont
    )

    return final_text

# =========================
# Regex patterns
# =========================

DATE_RE = re.compile(r"^\s*(\d{2}[/-]\d{2}[/-]\d{4})\s+")
MONEY_RE = re.compile(r"[-+]?\s*(?:\d{1,3}(?:\s\d{3})+|\d+)(?:[.,]\d{2})")

CATEGORY_OPTIONS = [
    "Clothing & Shoes",
    "Home Maintenance",
    "Cash Withdrawal",
    "Other Income",
    "Takeaways",
    "Groceries",
    "Alcohol",
    "Transfer",
    "Cellphone",
    "Interest",
    "Fees",
    "Fuel",
]


# =========================
# Helper functions
# =========================

def parse_date(s: str):
    s = s.replace("-", "/")
    return datetime.strptime(s, "%d/%m/%Y").date()


def clean_number(s: str) -> float:
    s = s.replace(" ", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)


def rebuild_wrapped_lines(lines: Iterable[str]) -> list[str]:
    rebuilt = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if DATE_RE.match(line):
            if buffer:
                rebuilt.append(buffer)
            buffer = line
        else:
            buffer += " " + line

    if buffer:
        rebuilt.append(buffer)

    return rebuilt


def extract_category(text_part: str, categories: Optional[list[str]] = None) -> tuple[str, str]:
    categories = categories or CATEGORY_OPTIONS
    trimmed = text_part.strip()

    for category in sorted(categories, key=len, reverse=True):
        if trimmed.endswith(category):
            description = trimmed[: -len(category)].strip()
            return description, category

    parts = trimmed.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return trimmed, ""


def split_transaction(line: str, categories: Optional[list[str]] = None):
    date_match = DATE_RE.match(line)
    if not date_match:
        return None

    date = parse_date(date_match.group(1))
    remainder = line[date_match.end():].strip()

    numbers = MONEY_RE.findall(remainder)
    if len(numbers) < 2:
        return None

    numbers = [clean_number(n) for n in numbers]
    balance = numbers[-1]

    fee = 0.0
    amount_index = -2

    if len(numbers) >= 3 and abs(numbers[-2]) <= 200:
        fee = abs(numbers[-2])
        amount_index = -3

    amount = numbers[amount_index]

    money_in = 0.0
    money_out = 0.0

    if amount < 0:
        money_out = abs(amount)
    else:
        money_in = amount

    text_part = remainder
    for _ in range(2 + (1 if fee else 0)):
        text_part = re.sub(
            r"\s*[-+]?\s*(?:\d{1,3}(?:\s\d{3})+|\d+)(?:[.,]\d{2})\s*$",
            "",
            text_part,
        )

    description, category = extract_category(text_part, categories)

    return {
        "Date": date,
        "Description": description,
        "Category": category,
        "Money In": money_in,
        "Money Out": money_out,
        "Fee": fee,
        "Balance": balance,
    }


# =========================
# Parsing
# =========================

def parse_transactions(raw_text: str, categories: Optional[list[str]] = None) -> pd.DataFrame:
    lines = raw_text.splitlines()
    rebuilt_lines = rebuild_wrapped_lines(lines)

    rows = []
    for line in rebuilt_lines:
        tx = split_transaction(line, categories)
        if tx:
            rows.append(tx)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    numeric_cols = ["Money In", "Money Out", "Fee", "Balance"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# =========================
# MONTHLY SUMMARY (KEY PART)
# =========================

def summarize_monthly_money_in(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Month", "Month Label", "Sum of Money In"])

    summary = df.copy()

    # 1️⃣ Ensure datetime
    summary["Date"] = pd.to_datetime(summary["Date"])

    # 2️⃣ Convert to monthly period (time-aware, sortable)
    summary["Month"] = summary["Date"].dt.to_period("M")

    # 3️⃣ Group + sum + FORCE chronological order
    monthly = (
        summary.groupby("Month", as_index=False)["Money In"]
        .sum()
        .sort_values("Month")
    )

    # 4️⃣ Convert Period → Timestamp (month start for plotting)
    monthly["Month"] = monthly["Month"].dt.to_timestamp(how="start")

    # 5️⃣ Human-readable label (NOT for sorting)
    monthly["Month Label"] = monthly["Month"].dt.strftime("%b-%y")

    monthly = monthly.rename(columns={"Money In": "Sum of Money In"})

    return monthly


# =========================
# Streamlit App
# =========================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Capitec Statement Cleaner", layout="wide")

    st.title("Capitec Statement Cleaner")
    st.write(
        "Upload a Capitec statement PDF file or paste the statement text below to "
        "convert it into a clean, downloadable table."
    )

    # -------------------------
    # Init session state
    # -------------------------
    if "df" not in st.session_state:
        st.session_state.df = None

    uploaded_file = st.file_uploader("Upload Capitec PDF statement", type=["pdf"])
    paste_text = st.text_area("Or paste statement text", height=220)

    custom_categories = st.text_input(
        "Optional: override categories (comma-separated)",
        value=", ".join(CATEGORY_OPTIONS),
    )

    categories = [c.strip() for c in custom_categories.split(",") if c.strip()]

    raw_text = ""
    if paste_text.strip():
        raw_text = paste_text
        
    elif uploaded_file is not None:
        raw_text = capitec_pdf_to_clean_text(uploaded_file)


    # -------------------------
    # Process button
    # -------------------------
    if st.button("Process statement", type="primary", disabled=not raw_text.strip()):
        df = parse_transactions(raw_text, categories=categories)

        if df.empty:
            st.warning("No transactions were detected. Check the input text formatting.")
        else:
            st.session_state.df = df

    # -------------------------
    # Render results (PERSISTENT)
    # -------------------------
    if st.session_state.df is not None:
        df = st.session_state.df

        st.success(f"Processed {len(df)} transactions.")
        st.dataframe(df, width='stretch')

        monthly_summary = summarize_monthly_money_in(df)

        # ✅ VISIBILITY TOGGLE (NOW SAFE)
        show_trend = st.checkbox("Show monthly income trend", value=True)

        if show_trend and not monthly_summary.empty:
            st.subheader("Monthly Money In Trend")

            trend_data = (
                monthly_summary
                .set_index("Month")[["Sum of Money In"]]
            )

            st.line_chart(trend_data)

            st.dataframe(
                monthly_summary[["Month Label", "Sum of Money In"]],
                width='stretch',
            )

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="capitec_statement_cleaned.csv",
            mime="text/csv",
        )

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Transactions")

        st.download_button(
            "Download Excel",
            data=excel_buffer.getvalue(),
            file_name="capitec_statement_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )



if __name__ == "__main__":
    run_streamlit_app()


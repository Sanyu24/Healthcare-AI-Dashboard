"""
=============================================================================
 app.py
=============================================================================
STEP 7 OF THE PIPELINE: THE DASHBOARD (main application entry point)

This is the file you actually RUN to launch Project Walter:
    streamlit run src/app.py

This file does not contain any data-cleaning, analysis, charting, or AI
logic itself -- it IMPORTS and ORCHESTRATES the six modules we already
built (data.py, cleanData.py, analyzeData.py, graph.py, outlierDetect.py,
report.py). Its only job is to:
    1. Build the user interface (sidebar + main page layout)
    2. Call the right function from the right module at the right time
    3. Display each function's result on screen

WHY THIS SEPARATION MATTERS:
Keeping all of the actual "thinking" (cleaning, math, charting, AI calls)
in their own separate files means this dashboard file stays short and
readable, and means every one of those modules can be tested completely
on its own (as we did in each file's standalone test block) without ever
needing Streamlit running at all.

DASHBOARD LAYOUT:
    SIDEBAR:
        - Excel file upload
        - Column mapping (auto-detected, user can override)
        - Filters: Year, Month, Supplier, Market, Cost Center, Account
    MAIN PAGE:
        - KPI cards: Total Spend, Largest Purchase, Average Purchase,
          Number of Suppliers, Number of Outliers
        - Charts: Monthly trend, Supplier, Market, Cost Center, Account,
          Outlier scatter
        - Claude-generated Executive Summary (6 sections)
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# streamlit is the library that turns this plain Python script into an
# interactive web dashboard -- every st.xxx() call below adds one visible
# piece of the page (a button, a chart, a table, etc.).
import streamlit as st

import pandas as pd

# Import every function we actually need from each of our six pipeline
# modules. Because app.py lives in the same src/ folder as these files,
# Python can import them directly by name.
from data import load_excel_file, list_sheet_names
from cleanData import clean_data, aggregate_purchase_orders
from analyzeData import (
    monthly_spending,
    supplier_spending,
    market_spending,
    cost_center_spending,
    account_spending,
    average_purchase_amount,
    largest_purchase,
    number_of_suppliers,
)
from outlierDetect import detect_outliers_per_market
from graph import (
    monthly_spending_line_chart,
    supplier_spending_bar_chart,
    market_spending_pie_chart,
    cost_center_spending_bar_chart,
    account_spending_pie_chart,
    outlier_scatter_plot,
)
from report import generate_executive_report, ReportGenerationError, save_report_to_file


# =============================================================================
# SECTION 1: PAGE CONFIGURATION
# =============================================================================
# st.set_page_config() must be the FIRST Streamlit command in the script.
# It controls the browser tab title/icon and, importantly, "wide" layout,
# which gives the dashboard the full width of the screen -- essential for
# a data-heavy app like this one instead of the default narrow, centered
# column.
# =============================================================================
st.set_page_config(
    page_title="Project Walter | Clinical Engineering Artificial Intelligence",
    page_icon="🏥",
    layout="wide",
)


# =============================================================================
# SECTION 2: SMALL HELPER FUNCTION -- AUTOMATIC COLUMN DETECTION
# =============================================================================
# Real hospital Excel exports name their columns differently ("PO Date" vs
# "Order Date", "Supplier" vs "Vendor Name"). Every module we built
# (cleanData.py, analyzeData.py, etc.) needs to be told the EXACT column
# name to use. Rather than making the user type these in from scratch,
# this helper makes an educated guess by searching column names for
# common keywords, and the sidebar lets the user CONFIRM or CORRECT that
# guess before anything is processed.
# =============================================================================
def guess_column(columns: list, keywords: list) -> str:
    """
    Guess the best matching column.
    Prefer Name over Number/Code/ID whenever possible.
    """

    priority_words = ["name", "description"]

    # 1. Exact match
    for keyword in keywords:
        for column in columns:
            if column.lower() == keyword.lower():
                return column

    # 2. Prefer "Name"
    for keyword in keywords:
        for priority in priority_words:
            for column in columns:
                lower = column.lower()
                if keyword in lower and priority in lower:
                    return column

    # 3. Ignore IDs/Codes/Numbers
    for keyword in keywords:
        for column in columns:
            lower = column.lower()

            if (
                keyword in lower
                and "number" not in lower
                and "code" not in lower
                and "id" not in lower
            ):
                return column

    # 4. Fallback
    for keyword in keywords:
        for column in columns:
            if keyword in column.lower():
                return column

    return None


# =============================================================================
# SECTION 3: CACHED DATA LOADING + CLEANING
# =============================================================================
# @st.cache_data tells Streamlit: "if this function is called again with
# the EXACT same inputs, don't re-run it -- just reuse the result from
# last time." This matters a lot here because Streamlit re-runs the
# ENTIRE script from top to bottom every single time the user interacts
# with any widget (like changing a filter dropdown). Without caching,
# we'd re-read and re-clean the whole Excel file on every tiny click,
# which would make the app feel slow. The underscore prefix on the
# uploaded_file parameter name tells Streamlit "don't try to use this
# specific argument as part of the cache key" -- we handle that ourselves
# by hashing the file's raw bytes instead, which is more reliable.
# =============================================================================
@st.cache_data(show_spinner="Reading and cleaning your Excel file...")
def load_and_clean_data(file_bytes: bytes, sheet_name: str, date_col: str, amount_col: str, supplier_col: str):
    """
    Load the uploaded Excel file and run it through the full cleaning
    pipeline. Cached so this expensive work only happens once per unique
    combination of file + column mapping.

    Parameters
    ----------
    file_bytes : bytes
        The raw bytes of the uploaded Excel file (used as the cache key).
    sheet_name : str
        Which sheet to load.
    date_col, amount_col, supplier_col : str
        The user-confirmed column names needed by clean_data().

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        The cleaned DataFrame and the cleaning report dictionary.
    """
    # io.BytesIO wraps raw bytes so pandas can read them as if they were
    # a real file on disk -- necessary because Streamlit gives us the
    # uploaded file's bytes, not a file path.
    import io
    raw_df = load_excel_file(io.BytesIO(file_bytes), sheet_name=sheet_name)
    cleaned_df, cleaning_report = clean_data(raw_df, date_col, amount_col, supplier_col)
    return cleaned_df, cleaning_report


# =============================================================================
# SECTION 4: SIDEBAR -- FILE UPLOAD
# =============================================================================
# st.sidebar.xxx() places a widget in the left-hand sidebar instead of the
# main page. This keeps setup/configuration controls (upload, filters)
# visually separate from the actual report content in the main area.
# =============================================================================
st.sidebar.title("📁 Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a procurement Excel workbook (.xlsx)",
    type=["xlsx"],
    help="Your file is processed only in this session and is not stored permanently.",
)

# If nothing has been uploaded yet, show a friendly welcome message on
# the main page and stop the script here (st.stop() halts execution of
# everything below it) -- there's nothing else useful to show yet.
if uploaded_file is None:
    st.title(" Project Walter")
    st.subheader("Clinical Engineering Artificial Intelligence Agent")
    st.info(
        "👈 Upload a Excel workbook using the sidebar to get started. "
        "This dashboard will automatically clean your data, analyze spending "
        "patterns, flag unusual purchases, and generate an AI-written executive "
        "summary."
    )
    st.stop()

# Read the sheet names so the user can pick the right one if the workbook
# has more than one tab.
file_bytes = uploaded_file.getvalue()
import io
sheet_names = list_sheet_names(io.BytesIO(file_bytes))
selected_sheet = st.sidebar.selectbox("Select worksheet", options=sheet_names, index=0)


# =============================================================================
# SECTION 5: SIDEBAR -- COLUMN MAPPING (auto-detected, user-confirmable)
# =============================================================================
# We peek at the raw column names (without fully loading/cleaning yet) so
# we can offer smart default guesses in each dropdown below.
# =============================================================================
preview_df = load_excel_file(io.BytesIO(file_bytes), sheet_name=selected_sheet)
all_columns = list(preview_df.columns)

with st.sidebar.expander("⚙️ Column Mapping (auto-detected — confirm or fix)", expanded=False):
    st.caption("We guessed these from your column names. Change any that look wrong.")

    date_column = st.selectbox(
        "PO Date column",
        options=all_columns,
        index=all_columns.index(guess_column(all_columns, ["date"])) if guess_column(all_columns, ["date"]) else 0,
    )
    amount_column = st.selectbox(
        "PO Amount column",
        options=all_columns,
        index=all_columns.index(guess_column(all_columns, ["amount", "cost", "price", "total"]))
        if guess_column(all_columns, ["amount", "cost", "price", "total"]) else 0,
    )
    po_number_column = st.selectbox(
        "PO Number column",
        options=all_columns,
        index=all_columns.index(guess_column(all_columns, ["po number", "po #", "purchase order", "po"]))
        if guess_column(all_columns, ["po number", "po #", "purchase order", "po"]) else 0,
    )
    supplier_column = st.selectbox(
        "Supplier column",
        options=all_columns,
        index=all_columns.index(guess_column(all_columns, ["supplier", "vendor"]))
        if guess_column(all_columns, ["supplier", "vendor"]) else 0,
    )

    # Market, Cost Center, and Account are OPTIONAL -- not every hospital
    # export tracks these, so we include a "(None)" choice for each.
    optional_options = ["(None)"] + all_columns

    def _optional_index(keywords):
        guess = guess_column(all_columns, keywords)
        return optional_options.index(guess) if guess else 0

    market_column = st.selectbox("Market/Category column (optional)", options=optional_options, index=_optional_index(["market", "category"]))
    cost_center_column = st.selectbox("Cost Center column (optional)", options=optional_options, index=_optional_index(["cost center", "costcenter"]))
    account_column = st.selectbox("Account/GL column (optional)", options=optional_options, index=_optional_index(["account", "gl"]))

# Convert the "(None)" placeholder back into a real Python None, since
# that's what our analyzeData.py / outlierDetect.py functions expect for
# "this optional column wasn't provided."
market_column = None if market_column == "(None)" else market_column
cost_center_column = None if cost_center_column == "(None)" else cost_center_column
account_column = None if account_column == "(None)" else account_column


# =============================================================================
# SECTION 6: LOAD + CLEAN THE DATA (using our cached pipeline function)
# =============================================================================
cleaned_df, cleaning_report = load_and_clean_data(
    file_bytes, selected_sheet, date_column, amount_column, supplier_column
)


cleaned_df = aggregate_purchase_orders(
    cleaned_df,
    po_column=po_number_column,
    amount_column=amount_column,
    supplier_column=supplier_column,
    date_column=date_column,
    market_column=market_column,
    cost_center_column=cost_center_column,
    account_column=account_column,
)

st.write("Amount column selected:", amount_column)

st.write("Maximum value:", cleaned_df[amount_column].max())

st.write("Total spend:", cleaned_df[amount_column].sum())

st.dataframe(
    cleaned_df.nlargest(
        10,
        amount_column
    )[[po_number_column, amount_column]]
)

st.sidebar.success(
    f"✅ Loaded {cleaning_report['rows_before_cleaning']} rows → "
    f"{cleaning_report['rows_after_cleaning']} clean rows "
    f"({cleaning_report['duplicate_rows_removed']} duplicates removed)"
)


# =============================================================================
# SECTION 7: SIDEBAR -- FILTERS
# =============================================================================
# Every filter widget below follows the same pattern: get the unique
# values available in the cleaned data, let the user pick a subset (with
# "select all" as the default so nothing is hidden until the user
# actively chooses to narrow things down), and skip the widget entirely
# if that column isn't present in this dataset.
# =============================================================================
st.sidebar.title("🔍 Filters")

# --- Year filter ---
available_years = sorted(cleaned_df["Year"].dropna().unique().tolist())
selected_years = st.sidebar.multiselect("Year", options=available_years, default=available_years)

# --- Month filter ---
available_months = sorted(cleaned_df["Month"].dropna().unique().tolist())
selected_months = st.sidebar.multiselect("Month", options=available_months, default=available_months)

# --- Supplier filter ---
available_suppliers = sorted(cleaned_df[supplier_column].dropna().unique().tolist())
selected_suppliers = st.sidebar.multiselect("Supplier", options=available_suppliers, default=available_suppliers)

# --- Market filter (only shown if this column exists) ---
selected_markets = None
if market_column:
    available_markets = sorted(cleaned_df[market_column].dropna().unique().tolist())
    selected_markets = st.sidebar.multiselect("Market", options=available_markets, default=available_markets)

# --- Cost Center filter (only shown if this column exists) ---
selected_cost_centers = None
if cost_center_column:
    available_cost_centers = sorted(cleaned_df[cost_center_column].dropna().unique().tolist())
    selected_cost_centers = st.sidebar.multiselect("Cost Center", options=available_cost_centers, default=available_cost_centers)

# --- Account filter (only shown if this column exists) ---
selected_accounts = None
if account_column:
    available_accounts = sorted(cleaned_df[account_column].dropna().unique().tolist())
    selected_accounts = st.sidebar.multiselect("Account", options=available_accounts, default=available_accounts)


# =============================================================================
# SECTION 8: APPLY FILTERS
# =============================================================================
# IMPORTANT DESIGN NOTE: filters are applied BEFORE any analysis, charting,
# or outlier detection happens below -- not just before display. This
# means if a user filters down to a single supplier, outlier detection
# recalculates what "normal" looks like specifically WITHIN that
# supplier's own spending pattern, rather than always comparing against
# the whole organization. This makes filtered views genuinely useful for
# focused investigation, not just a visual crop of the same numbers.
# =============================================================================
filtered_df = cleaned_df[
    cleaned_df["Year"].isin(selected_years)
    & cleaned_df["Month"].isin(selected_months)
    & cleaned_df[supplier_column].isin(selected_suppliers)
].copy()

if market_column and selected_markets is not None:
    filtered_df = filtered_df[filtered_df[market_column].isin(selected_markets)]
if cost_center_column and selected_cost_centers is not None:
    filtered_df = filtered_df[filtered_df[cost_center_column].isin(selected_cost_centers)]
if account_column and selected_accounts is not None:
    filtered_df = filtered_df[filtered_df[account_column].isin(selected_accounts)]

# Defensive check: if the filters have narrowed the data down to nothing,
# tell the user clearly rather than letting every chart below silently
# fail or render empty and confusing.
if filtered_df.empty:
    st.warning("No purchase orders match the current filter selection. Try widening your filters.")
    st.stop()


# =============================================================================
# SECTION 9: RUN THE ANALYSIS + OUTLIER DETECTION ON THE FILTERED DATA
# =============================================================================
monthly_df = monthly_spending(filtered_df, amount_column)
supplier_df = supplier_spending(filtered_df, amount_column, supplier_column)
market_df = market_spending(filtered_df, amount_column, market_column) if market_column else pd.DataFrame()
cost_center_df = cost_center_spending(filtered_df, amount_column, cost_center_column) if cost_center_column else pd.DataFrame()
account_df = account_spending(filtered_df, amount_column, account_column) if account_column else pd.DataFrame()

# ---------- Keep only Top 10 ----------
supplier_df = (
    supplier_df
    .sort_values("Total Spend", ascending=False)
    .head(10)
)

if not market_df.empty:
    market_df = (
        market_df
        .sort_values("Total Spend", ascending=False)
        .head(10)
    )

if not cost_center_df.empty:
    cost_center_df = (
        cost_center_df
        .sort_values("Total Spend", ascending=False)
        .head(10)
    )

normal_df, outlier_df = detect_outliers_per_market(
    filtered_df,
    po_column=po_number_column,
    supplier_column=supplier_column,
    amount_column=amount_column,
    market_column=market_column,
    cost_center_column=cost_center_column,
)

# Build an "Is Outlier" True/False column on the filtered data (matched
# by PO Number) so outlier_scatter_plot() can color-code flagged
# purchases directly on the full timeline.
filtered_df["Is Outlier"] =  False

if not outlier_df.empty:
    filtered_df.loc[
        filtered_df.index.isin(outlier_df.index),
        "Is Outlier"
    ] = True


# =============================================================================
# SECTION 10: MAIN PAGE HEADER
# =============================================================================
st.title("🏥 Project Walter — Artificial Intelligence Dashboard")
st.caption("Clinical Engineering Department | Cleveland Clinic")
st.divider()


# =============================================================================
# SECTION 11: KPI CARDS
# =============================================================================
# st.columns(5) creates 5 equal-width side-by-side containers. st.metric()
# renders a large, clean "headline number" style card -- this is the
# standard way to show top-line KPIs at the very top of a professional
# dashboard, before any charts, so an executive can see the big picture
# in the first three seconds of looking at the page.
# =============================================================================
st.subheader("📊 Key Performance Indicators")

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

total_spend = filtered_df[amount_column].sum()
avg_purchase = average_purchase_amount(filtered_df, amount_column).iloc[0, 0]
largest_row = largest_purchase(filtered_df, amount_column)
largest_amount = largest_row.iloc[0][amount_column]
supplier_count = number_of_suppliers(filtered_df, supplier_column).iloc[0, 0]
outlier_count = len(outlier_df)

kpi_col1.metric("Total Spending", f"${total_spend:,.0f}")
kpi_col2.metric("Largest Purchase", f"${largest_amount:,.0f}")
kpi_col3.metric("Average Purchase", f"${avg_purchase:,.0f}")
kpi_col4.metric("Number of Suppliers", f"{supplier_count}")
kpi_col5.metric("Purchases Requiring Review", f"{outlier_count}", delta_color="inverse")

st.divider()


# =============================================================================
# SECTION 12: CHARTS
# =============================================================================
# Each chart is built by calling the matching function from graph.py, and
# displayed using st.plotly_chart(). use_container_width=True makes each
# chart stretch to fill its column, which looks far more professional
# than Plotly's small fixed-size default.
# =============================================================================
st.subheader("📈 Spending Analysis")

# --- Monthly trend: full width, since a time trend benefits from more
# horizontal space to show change clearly ---
st.plotly_chart(monthly_spending_line_chart(monthly_df), use_container_width=True)

# --- Supplier + Market side by side ---
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(supplier_spending_bar_chart(supplier_df, supplier_column), use_container_width=True)
with chart_col2:
    if market_column:
        st.plotly_chart(market_spending_pie_chart(market_df, market_column), use_container_width=True)
    else:
        st.info("No Market/Category column was mapped for this dataset.")

# --- Cost Center + Account side by side ---
chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    if cost_center_column:
        st.plotly_chart(cost_center_spending_bar_chart(cost_center_df, cost_center_column), use_container_width=True)
    else:
        st.info("No lumn was mapped for this dataset.")


# --- Outlier scatter plot: full width, since individual points over time
# benefit from more horizontal room to spread out and stay readable ---
st.plotly_chart(
    outlier_scatter_plot(filtered_df, date_column, amount_column, outlier_column="Is Outlier"),
    use_container_width=True,
)

if not outlier_df.empty:

    st.subheader("🚨 Purchases Requiring Review")

    
    top_outliers = (
        outlier_df
        .drop_duplicates(subset="PO Number")
        .sort_values("Amount", ascending=False)
        .head(25)
    )

    st.caption(
        "Showing the 25 highest-value purchases flagged for review."
    )

    st.dataframe(
        top_outliers,
        use_container_width=True,
    )

st.divider()


# =============================================================================
# SECTION 13: CLAUDE-GENERATED EXECUTIVE SUMMARY
# =============================================================================
# This section is only triggered by a button click (not run automatically
# on every filter change) since it costs an API call each time -- we
# don't want to accidentally send a request to Claude every time someone
# adjusts a filter dropdown.
# =============================================================================
st.subheader("🤖 AI-Generated Executive Summary")
st.caption("Powered by Claude — analyzes the summary tables above, not your raw spreadsheet.")

if st.button("Generate Executive Summary", type="primary"):
    with st.spinner("Claude is analyzing your procurement data..."):
        try:
            result = generate_executive_report(
                monthly_df,
                supplier_df,
                market_df if market_column else pd.DataFrame(),
                outlier_df,
                cost_center_df if cost_center_column else pd.DataFrame(),
            )

            # Save the generated report as a file inside reports/, giving
            # the department a permanent, shareable record.
            saved_path = save_report_to_file(result["full_text"], "../reports/executive_summary.md")

            # st.tabs() creates a clean, clickable tab bar -- much more
            # organized than dumping all six sections in one long scroll.
            tab_names = [
                "Executive Summary", "Key Trends", "Top Suppliers",
                "Potential Risks", "Interesting Insights", "Recommendations",
            ]
            tabs = st.tabs(tab_names)
            for tab, section_name in zip(tabs, tab_names):
                with tab:
                    section_text = result["sections"].get(section_name, "")
                    st.markdown(section_text if section_text else "_No content returned for this section._")

            st.download_button(
                "📥 Download Full Report (.md)",
                data=result["full_text"],
                file_name="executive_summary.md",
                mime="text/markdown",
            )
            st.caption(f"Report also saved to: {saved_path}")

        except ReportGenerationError as error:
            # Our custom exception from report.py -- always a clear,
            # friendly message, never a raw/confusing traceback.
            st.error(f"Could not generate the executive summary: {error}")
else:
    st.info("Click the button above to have Claude write a plain-English executive report from your current filtered data.")

"""
=============================================================================
 cleanData.py
=============================================================================
STEP 2 OF THE PIPELINE: DATA CLEANING

This module takes the RAW DataFrame produced by data.py and turns it into a
CLEAN, ML-ready DataFrame. "ML-ready" means: correct data types, no
duplicate rows, no inconsistent text, and no unexplained missing values.

WHY CLEANING IS ITS OWN SEPARATE FILE:
Real hospital procurement exports are messy -- vendor names are typed
inconsistently ("Medtronic" vs "MEDTRONIC INC" vs " medtronic "), amount
columns often contain "$" and commas as plain text instead of numbers, and
dates can be stored as text. If we let analyzeData.py or graph.py deal with
this mess directly, every future feature we build would have to repeat the
same defensive cleaning logic. By cleaning ONCE here, every downstream
module (analysis, outlier detection, graphs, AI report) can trust the data
completely and focus only on its own job.

WHAT THIS FILE PROVIDES (public functions):
    1. strip_whitespace(df)                    -> trims stray spaces everywhere
    2. standardize_supplier_names(df, col)     -> normalizes vendor name text
    3. convert_amount_to_float(df, col)        -> turns "$4,500.00" into 4500.0
    4. convert_date_to_datetime(df, col)       -> turns date text into real dates
    5. handle_missing_values(df, ...)          -> fills/drops missing data
    6. remove_duplicate_rows(df, subset)       -> removes exact duplicate rows
    7. create_date_features(df, date_col)      -> adds Month, Quarter, Year, Week
    8. clean_data(df, ...)                     -> orchestrator: runs everything
                                                   in the correct order and
                                                   returns (clean_df, report)
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# pandas gives us the DataFrame object and all the vectorized operations
# (e.g. .str.strip(), .fillna(), .drop_duplicates()) that make cleaning an
# entire column fast, without writing a manual for-loop over every row.
import pandas as pd

# numpy is used for numeric helpers, most notably np.nan, which is the
# standard "this value is missing" marker pandas uses internally.
import numpy as np

# 're' (regular expressions) is used to strip out currency symbols and any
# non-numeric junk characters from the "PO Amount" column, e.g. turning the
# text "$4,500.00" into "4500.00" before converting it to a real float.
import re

# Type hints -- purely for readability, they don't change how the code runs.
from typing import List, Optional, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# FUNCTION 1: strip_whitespace
# ---------------------------------------------------------------------------
def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading/trailing whitespace from every text column AND from the
    column names themselves.

    WHY THIS MATTERS FOR ML / ANALYSIS:
    To a human, "Medtronic" and "Medtronic " (trailing space) look identical.
    To a computer, they are two completely different strings. Left
    uncleaned, this single invisible space will:
        - make groupby("Vendor") treat them as two different vendors
        - make drop_duplicates() miss rows that are otherwise identical
        - make a join/merge against another table silently fail to match
    This is one of the most common silent bugs in real-world data pipelines,
    so we fix it FIRST, before any other cleaning step runs.

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or partially cleaned) DataFrame.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with whitespace trimmed from column names and from
        every text (object/string) column's values.
    """

    # We work on a COPY of the DataFrame, never the original. This is a
    # defensive habit: it prevents this function from accidentally
    # modifying a DataFrame that the caller (or another part of the
    # program) still has a reference to and expects to be unchanged.
    df = df.copy()

    # Strip whitespace from the column NAMES themselves, e.g. a column
    # literally named " Total Cost " becomes "Total Cost". We rebuild the
    # .columns list using a list comprehension that calls .strip() on each
    # name (only if it's a string -- some columns could theoretically have
    # non-string names).
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

    # select_dtypes(include="object") grabs only the columns pandas is
    # storing as generic text/mixed-type ("object") columns -- this is how
    # pandas represents normal string columns like "Vendor" or "Item
    # Description". We don't want to touch numeric or datetime columns
    # here, since .str operations don't apply to them.
    text_columns = df.select_dtypes(include=["object", "string"]).columns

    # Loop over just those text columns and clean each one.
    for col in text_columns:
        # .str.strip() removes leading/trailing whitespace from every
        # value in the column, all at once (vectorized -- much faster than
        # looping row by row in plain Python).
        # We first convert to string with .astype(str) to protect against
        # a column that mixes real strings with other types (e.g. NaN),
        # then immediately turn any resulting literal "nan" text back into
        # a true missing value with .replace(), so we don't accidentally
        # create the fake text "nan" where a real missing value used to be.
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )

    return df


# ---------------------------------------------------------------------------
# FUNCTION 2: standardize_supplier_names
# ---------------------------------------------------------------------------
def standardize_supplier_names(df: pd.DataFrame, supplier_column: str) -> pd.DataFrame:
    """
    Normalize vendor/supplier name text so the same real-world company is
    always represented by exactly one consistent string.

    WHY THIS MATTERS FOR ML / ANALYSIS:
    Procurement data almost always has inconsistent vendor names because
    different staff typed them in over the years: "medtronic", "MEDTRONIC",
    "Medtronic Inc.", "Medtronic, Inc". If we don't standardize these, any
    vendor-level analysis (total spend per vendor, top 10 vendors, outlier
    detection *within* a vendor's normal spending range) will incorrectly
    treat one real vendor as 3-4 separate, smaller vendors. This makes the
    "Top Vendor by Spend" ranking wrong, dilutes real vendor risk signals,
    and produces a misleading executive summary.

    What this function does, step by step:
        1. Converts everything to title case ("MEDTRONIC INC" -> "Medtronic Inc")
        2. Removes common corporate suffixes/punctuation noise
           (".", ",", "Inc", "LLC", "Co") so near-identical vendor names collapse
           into one clean form.
        3. Collapses any repeated internal spaces into a single space.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame (ideally already whitespace-stripped).
    supplier_column : str
        The name of the column containing supplier/vendor names.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the supplier column standardized.
    """

    df = df.copy()

    # Defensive check: if the caller passes a column name that doesn't
    # actually exist in this dataset (e.g. a typo, or a differently-named
    # export), we fail with a clear message rather than crashing later with
    # a confusing KeyError deep inside pandas.
    if supplier_column not in df.columns:
        raise ValueError(
            f"Column '{supplier_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # A small helper function that cleans ONE supplier name string. We
    # define it locally (nested) because it's only ever used inside this
    # function, right below, via .apply().
    def _clean_single_name(name: Any) -> Any:
        # If the value is already missing (NaN), leave it as NaN -- don't
        # try to run string operations on a missing value.
        if pd.isna(name):
            return name

        # Make sure we're working with a plain string.
        name = str(name)

        # Remove common corporate suffixes and punctuation that cause the
        # SAME company to appear as different strings. We use a regular
        # expression with the IGNORECASE flag so "inc", "Inc", "INC" are
        # all matched the same way. The "\b" marks a word boundary, so we
        # don't accidentally chop letters out of the middle of a real word.
        name = re.sub(r"[.,]", "", name)  # remove periods and commas
        name = re.sub(r"\b(Inc|LLC|Co|Corp|Ltd)\b", "", name, flags=re.IGNORECASE)

        # Collapse any run of multiple spaces (which the suffix removal
        # above may have left behind) down to a single space, then strip
        # any leading/trailing space that's left.
        name = re.sub(r"\s+", " ", name).strip()

        # .title() capitalizes the first letter of every word, so
        # "medtronic" and "MEDTRONIC" both become "Medtronic".
        return name.title()

    # .apply() runs our helper function on every single value in the
    # supplier column, one at a time, and replaces the column with the
    # cleaned results.
    df[supplier_column] = df[supplier_column].apply(_clean_single_name)

    return df


# ---------------------------------------------------------------------------
# FUNCTION 3: convert_amount_to_float
# ---------------------------------------------------------------------------
def convert_amount_to_float(df: pd.DataFrame, amount_column: str) -> pd.DataFrame:
    """
    Convert a spending/amount column (which may be stored as text, e.g.
    "$4,500.00") into a true numeric float column.

    WHY THIS MATTERS FOR ML / ANALYSIS:
    Excel exports frequently store money as formatted TEXT rather than as
    numbers -- especially when the column includes a "$" sign or thousands
    separator commas. As long as the column is text, pandas cannot sum it,
    average it, chart it, or feed it into any statistical/ML calculation.
    Every later step in this pipeline (total spend, monthly trend charts,
    outlier detection using z-scores/IQR) mathematically REQUIRES this
    column to be a real float. Skipping this step would cause silent
    failures or completely wrong totals (e.g. pandas might join "4500" and
    "600" as the text "4500600" instead of adding them to get 5100).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the amount column.
    amount_column : str
        Name of the column to convert, e.g. "PO Amount Ordered".

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the amount column converted to float64.
        Any value that cannot be converted becomes NaN (handled later by
        handle_missing_values), rather than crashing the whole pipeline.
    """

    df = df.copy()

    if amount_column not in df.columns:
        raise ValueError(
            f"Column '{amount_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Step A: Convert everything to string first so we can safely run text
    # cleanup on it, even if some cells are already numbers.
    cleaned_series = df[amount_column].astype(str)

    # Step B: Remove currency symbols, commas, and any stray spaces.
    # re.sub(pattern, replacement, string) replaces every match of pattern
    # with replacement. Here we replace any character that is NOT a digit,
    # a minus sign, or a decimal point with nothing (i.e. delete it).
    # We use .apply() combined with a lambda (a small one-line anonymous
    # function) to run this on every value in the column.
    cleaned_series = cleaned_series.apply(
        lambda value: re.sub(r"[^0-9\.\-]", "", value)
    )

    # Step C: Some cells might now be an empty string (e.g. the original
    # cell was blank or contained only a currency symbol). pd.to_numeric
    # can't convert an empty string to a number, so we explicitly replace
    # empty strings with NaN first.
    cleaned_series = cleaned_series.replace("", np.nan)

    # Step D: Convert the cleaned text into real numbers.
    # errors="coerce" means: if a value STILL can't be converted to a
    # number after our cleanup (e.g. it was garbage text like "N/A"),
    # turn it into NaN instead of crashing the entire program. This keeps
    # the pipeline running and lets handle_missing_values() deal with it
    # in one consistent place, later.
    df[amount_column] = pd.to_numeric(cleaned_series, errors="coerce")

    return df


# ---------------------------------------------------------------------------
# FUNCTION 4: convert_date_to_datetime
# ---------------------------------------------------------------------------
def convert_date_to_datetime(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Convert a date column (which may be stored as text, or inconsistent
    date formats) into pandas' real datetime64 type.

    WHY THIS MATTERS FOR ML / ANALYSIS:
    A "date" that is secretly stored as text (e.g. the string "01/05/2024")
    cannot be sorted chronologically, cannot have "3 months ago" style
    comparisons done on it, and cannot be used to extract calendar features
    like Month or Quarter (see create_date_features below). Time-series
    trend charts and seasonality analysis -- both central to procurement
    spend analysis -- are only possible once this column is a true
    datetime type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the date column.
    date_column : str
        Name of the column to convert, e.g. "PO Date".

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the date column converted to pandas datetime.
        Unparseable values become NaT ("Not a Time", the datetime
        equivalent of NaN) instead of crashing the pipeline.
    """

    df = df.copy()

    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # pd.to_datetime() understands a huge variety of date formats
    # automatically (e.g. "2024-01-05", "01/05/2024", "Jan 5, 2024").
    # errors="coerce" again means: anything it truly cannot understand
    # becomes NaT (missing date) instead of stopping the whole program.
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# FUNCTION 5: handle_missing_values
# ---------------------------------------------------------------------------
def handle_missing_values(
    df: pd.DataFrame,
    critical_columns: Optional[List[str]] = None,
    fill_text_with: str = "Unknown",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Decide, explicitly and transparently, what to do with missing values.

    WHY THIS MATTERS FOR ML / ANALYSIS:
    Almost every statistical calculation and ML algorithm either breaks or
    silently produces wrong results when it encounters NaN. But the WRONG
    way to handle this is to silently drop every row with any missing
    value at all -- that can throw away large amounts of good data and can
    bias the dataset (e.g. if one vendor's rows happen to be missing a
    "Notes" field more often, dropping those rows under-represents that
    vendor in the analysis). The right approach, used here, is:
        1. Rows missing a CRITICAL field (something we truly cannot
           analyze without, like the spend amount or the date) are
           dropped, because there is no reasonable value to guess for them.
        2. Rows missing a non-critical TEXT field (like Vendor or Item
           Description) are filled with a clear placeholder ("Unknown")
           so the row is kept, and so this gap is visible/traceable in
           charts and summaries rather than hidden.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean (should already have types converted,
        i.e. run this AFTER convert_amount_to_float / convert_date_to_datetime,
        so that "missing" means a true NaN/NaT, not just unparsed text).
    critical_columns : list of str, optional
        Columns that MUST have a value for a row to be usable at all
        (e.g. ["PO Amount Ordered", "PO Date"]). Rows missing any of these
        are dropped. If None, no rows are dropped based on this rule.
    fill_text_with : str, default "Unknown"
        The placeholder text used to fill missing values in non-critical
        TEXT columns, so those rows are still kept for analysis.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        The cleaned DataFrame, and a small report dictionary describing
        exactly what was dropped/filled (for transparency/auditability --
        important in a hospital finance context where you may need to
        justify exactly what happened to the data).
    """

    df = df.copy()
    report: Dict[str, Any] = {}

    # --- Step A: Drop rows missing a CRITICAL column ---
    rows_before = len(df)
    if critical_columns:
        # Only check columns that actually exist in this dataset, in case
        # the caller listed a column name that isn't present.
        existing_critical_columns = [c for c in critical_columns if c in df.columns]

        # dropna(subset=[...]) removes any row where ANY of the listed
        # columns is missing (NaN/NaT). "how='any'" is the default and is
        # what we want here: even one missing critical field disqualifies
        # the row from reliable analysis.
        df = df.dropna(subset=existing_critical_columns, how="any")

    rows_after_critical_drop = len(df)
    report["rows_dropped_missing_critical_fields"] = rows_before - rows_after_critical_drop

    # --- Step B: Fill missing values in non-critical TEXT columns ---
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    fill_counts: Dict[str, int] = {}

    for col in text_columns:
        # Count how many values are missing in this column BEFORE filling,
        # purely so we can report it afterward.
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            # .fillna() replaces every NaN in this column with our chosen
            # placeholder text, keeping the row instead of deleting it.
            df[col] = df[col].fillna(fill_text_with)
            fill_counts[col] = missing_count

    report["text_values_filled_per_column"] = fill_counts

    return df, report


# ---------------------------------------------------------------------------
# FUNCTION 6: remove_duplicate_rows
# ---------------------------------------------------------------------------
def remove_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows from the dataset.

    WHY THIS MATTERS FOR ML / ANALYSIS:
    Duplicate rows are extremely common in real procurement exports (e.g.
    the same purchase order accidentally exported twice, or a row
    duplicated during a spreadsheet copy-paste). If left in the data:
        - Total spend figures are inflated (the same purchase is counted twice)
        - Average order size is skewed
        - Outlier detection thresholds (which are based on the overall
          distribution of spending) become distorted by artificially
          repeated data points
    This is why we run this step AFTER strip_whitespace() and
    standardize_supplier_names() -- otherwise two rows that are true
    duplicates except for a trailing space or inconsistent vendor casing
    would NOT be recognized as duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to de-duplicate.
    subset : list of str, optional
        Specific columns to consider when checking for duplicates (e.g.
        ["PO Number"] if a repeated PO Number alone means "duplicate row",
        even if some other column differs slightly). If None, a row must
        match EXACTLY across every column to be considered a duplicate.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        The de-duplicated DataFrame, and the number of rows that were
        removed (useful for the cleaning report / audit trail).
    """

    df = df.copy()

    rows_before = len(df)

    # drop_duplicates() keeps the FIRST occurrence of each duplicated row
    # by default, and removes every occurrence after that.
    # keep="first" is stated explicitly here (even though it's the
    # default) so the behavior is obvious to anyone reading this code.
    df = df.drop_duplicates(subset=subset, keep="first")

    rows_after = len(df)
    num_removed = rows_before - rows_after

    return df, num_removed


# ---------------------------------------------------------------------------
# FUNCTION 7: create_date_features
# ---------------------------------------------------------------------------
def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create four new columns derived from a datetime column: Year, Month,
    Quarter, and Week.

    WHY THIS MATTERS FOR ML / ANALYSIS:
    A raw datetime value like "2024-02-10" is hard to directly use for
    grouping/aggregating or as a machine learning feature. By breaking it
    into explicit calendar parts, we unlock:
        - Month / Quarter columns: enable "spend by month" and "spend by
          quarter" trend charts, and let us detect SEASONALITY (e.g.
          "Q1 always has higher medical device spend due to budget
          renewal") -- a very common and important pattern in procurement.
        - Year column: enables year-over-year comparisons.
        - Week column: enables finer-grained trend detection and can help
          spot short-term spikes (e.g. a single unusually expensive week)
          that monthly aggregation might smooth over and hide.
    These columns also make GroupBy operations in analyzeData.py and
    graph.py much simpler and more readable than repeatedly extracting
    these values from the raw date on the fly.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame, which must already have `date_column` converted to
        a real datetime type (run convert_date_to_datetime() first).
    date_column : str
        The name of the datetime column to extract features from.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with four additional columns: "Year", "Month",
        "Quarter", "Week".
    """

    df = df.copy()

    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Defensive check: if this column isn't actually a datetime type yet
    # (e.g. the caller forgot to run convert_date_to_datetime first), the
    # .dt accessor used below would fail. We catch that here with a clear,
    # actionable error message instead of a confusing pandas traceback.
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise TypeError(
            f"Column '{date_column}' is not a datetime type. "
            f"Run convert_date_to_datetime() on it before calling this function."
        )

    # The .dt accessor unlocks date-specific properties on a datetime
    # column. Each of these operates on the WHOLE column at once
    # (vectorized), extracting one calendar attribute per row.
    df["Year"] = df[date_column].dt.year          # e.g. 2024
    df["Month"] = df[date_column].dt.month         # e.g. 1-12
    df["Quarter"] = df[date_column].dt.quarter      # e.g. 1-4
    df["Week"] = df[date_column].dt.isocalendar().week  # e.g. 1-53 (ISO week number)

    return df


# ---------------------------------------------------------------------------
# FUNCTION 8: clean_data  (the orchestrator)
# ---------------------------------------------------------------------------
def clean_data(
    df: pd.DataFrame,
    date_column: str,
    amount_column: str,
    supplier_column: str,
    duplicate_subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the FULL cleaning pipeline, in the correct order, on a raw
    DataFrame, and return both the cleaned data and a report describing
    exactly what was changed.

    This is the ONE function that main.py (and analyzeData.py) should
    actually call -- it exists so that nobody has to remember the correct
    order of the 7 individual cleaning steps above, or repeat that logic
    in multiple places.

    THE ORDER, AND WHY IT MATTERS:
        1. strip_whitespace          - must run first: stray spaces would
                                        otherwise cause standardize_supplier_names
                                        and remove_duplicate_rows to miss matches.
        2. standardize_supplier_names - run before de-duplication, so
                                         "Medtronic" and "MEDTRONIC INC" are
                                         recognized as the same vendor.
        3. convert_amount_to_float    - run before handle_missing_values,
                                         so that unparseable text becomes a
                                         real, detectable NaN.
        4. convert_date_to_datetime   - same reasoning as amount conversion,
                                         and this must happen BEFORE
                                         create_date_features (step 7).
        5. handle_missing_values      - run after type conversion, so
                                         "missing" reflects true NaN/NaT
                                         values, not just unparsed text.
        6. remove_duplicate_rows      - run after cleanup/standardization,
                                         so near-duplicates are caught.
        7. create_date_features       - run last, since it depends on the
                                         date column already being a valid,
                                         cleaned datetime type.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame as loaded by data.py.
    date_column : str
        Name of the purchase-order date column, e.g. "PO Date".
    amount_column : str
        Name of the purchase-order amount column, e.g. "PO Amount Ordered".
    supplier_column : str
        Name of the vendor/supplier name column, e.g. "Supplier".
    duplicate_subset : list of str, optional
        Specific columns to check when removing duplicates. If None,
        entire rows must match exactly to be considered duplicates.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        The fully cleaned DataFrame, ready for analysis, and a report
        dictionary summarizing every change made -- useful for displaying
        a "Data Cleaning Summary" in the Streamlit UI, and for auditability.
    """

    # This dictionary will collect a running log of every cleaning
    # decision made, so the whole process is transparent, not a "black
    # box." This matters in a hospital finance context, where someone may
    # later ask "why does the total spend not match the raw Excel file?"
    cleaning_report: Dict[str, Any] = {}
    rows_at_start = len(df)

    # Step 1: whitespace cleanup, applied to the entire DataFrame.
    df = strip_whitespace(df)

    # Step 2: standardize the supplier/vendor name column.
    df = standardize_supplier_names(df, supplier_column)

    # Step 3: convert the amount column from text/mixed to float.
    df = convert_amount_to_float(df, amount_column)

    # Step 4: convert the date column from text/mixed to real datetime.
    df = convert_date_to_datetime(df, date_column)

    # Step 5: handle missing values. We treat the amount and date columns
    # as CRITICAL -- a purchase record with no cost or no date can't be
    # meaningfully analyzed, so those rows are dropped. Everything else
    # (like a missing item description) is filled with "Unknown" instead
    # of losing the whole row.
    df, missing_values_report = handle_missing_values(
        df,
        critical_columns=[amount_column, date_column],
    )
    cleaning_report["missing_values"] = missing_values_report

    # Step 6: remove duplicate rows (now that text is standardized).
    df, num_duplicates_removed = remove_duplicate_rows(df, subset=duplicate_subset)
    cleaning_report["duplicate_rows_removed"] = num_duplicates_removed

    # Step 7: create Year / Month / Quarter / Week columns from the now-
    # guaranteed-clean date column.
    df = create_date_features(df, date_column)

    # Final bookkeeping for the report: how many rows did we start with,
    # and how many survived the full cleaning process?
    cleaning_report["rows_before_cleaning"] = rows_at_start
    cleaning_report["rows_after_cleaning"] = len(df)
    cleaning_report["total_rows_removed"] = rows_at_start - len(df)

    return df, cleaning_report

def aggregate_purchase_orders(
    dataframe,
    po_column,
    amount_column,
    supplier_column,
    date_column,
    market_column=None,
    cost_center_column=None,
    account_column=None,
    requester_column=None,
):
    """
    Combine multiple line items belonging to the same Purchase Order
    into one Purchase Order summary.
    """

    group_columns = {}

    for column in dataframe.columns:

        if column == po_column:
            continue

        elif column == amount_column:
            group_columns[column] = "max"

        else:
            group_columns[column] = "first"
            
    aggregated = (
        dataframe
        .groupby(po_column, as_index=False)
        .agg(group_columns)
    )

    return aggregated

# ---------------------------------------------------------------------------
# STANDALONE TEST BLOCK
# ---------------------------------------------------------------------------
# Runs only when you execute this file directly:  python src/cleanData.py
# We build a deliberately MESSY sample dataset (duplicates, inconsistent
# vendor names, whitespace, currency-formatted text, and missing values)
# so we can visibly prove every cleaning function works correctly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # A deliberately messy sample dataset mimicking real-world procurement
    # export problems.
    messy_data = {
        "PO Number": ["PO-1001", "PO-1002", "PO-1003", "PO-1004", "PO-1004", "PO-1005", "PO-1006"],
        "Supplier": [
            " Medtronic", "GE Healthcare", "STRYKER INC.", "medtronic",
            "medtronic", "Philips Co.", None
        ],
        "Item Description": [
            "Infusion Pump", "Patient Monitor", "Surgical Drill",
            "Infusion Pump ", "Infusion Pump ", "Ultrasound Probe", "Ventilator Filter"
        ],
        "PO Amount Ordered": [
            "$4,500.00", "12800.50", "$3,200", "$4,600.00",
            "$4,600.00", "980.25", "N/A"
        ],
        "PO Date": [
            "2024-01-05", "2024-01-12", "2024-02-01",
            "2024-02-10", "2024-02-10", "2024-03-03", "2024-03-15"
        ],
    }
    messy_df = pd.DataFrame(messy_data)

    print("RAW (messy) DATA:")
    print(messy_df)
    print("\nRAW dtypes:")
    print(messy_df.dtypes)

    # Run the full cleaning pipeline in one call.
    cleaned_df, report = clean_data(
        messy_df,
        date_column="PO Date",
        amount_column="PO Amount Ordered",
        supplier_column="Supplier",
    )

    print("\n" + "=" * 60)
    print("CLEANED DATA:")
    print("=" * 60)
    print(cleaned_df)

    print("\nCLEANED dtypes:")
    print(cleaned_df.dtypes)

    print("\n" + "=" * 60)
    print("CLEANING REPORT:")
    print("=" * 60)
    for key, value in report.items():
        print(f"{key}: {value}")

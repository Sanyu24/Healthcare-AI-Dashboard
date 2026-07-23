"""
=============================================================================
 data.py
=============================================================================
STEP 1 OF THE PIPELINE: DATA LOADING

This is the very first module in the Project Walter pipeline. Its ONLY job
is to safely get an Excel workbook off the disk (or out of a Streamlit
file-uploader buffer) and turn it into a pandas DataFrame that the rest of
the application can work with.

This file deliberately does NOT:
    - clean or transform the data       (that happens in cleanData.py)
    - analyze spending                  (that happens in analyzeData.py)
    - detect outliers                   (that happens in outlierDetect.py)

Keeping "loading" completely separate from "cleaning/analysis" means that
if something breaks, we immediately know WHERE it broke:
    - If it breaks here            -> the file itself is the problem
    - If it breaks in cleanData.py -> the data's content is the problem

WHAT THIS FILE PROVIDES (public functions other files/main.py will call):
    1. list_sheet_names(file)          -> peek at sheet names in a workbook
    2. load_excel_file(file, sheet)    -> load a sheet into a DataFrame
    3. validate_file_exists(path)      -> confirm a file path is real/valid
    4. validate_required_columns(...)  -> confirm expected columns are present
    5. get_basic_file_info(df)         -> rows, columns, dtypes, missing data
    6. display_data_summary(df)        -> nicely prints everything requested:
                                             - number of rows
                                             - number of columns
                                             - column names
                                             - data types
                                             - missing values
                                             - first five rows
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# pandas is THE core library for tabular data in Python. We use it to read
# Excel files and represent the dataset as a "DataFrame" (think: a smart
# spreadsheet object in memory).
import pandas as pd

# 'os' lets us check whether a file actually exists on disk, and lets us
# inspect file extensions (e.g. making sure someone didn't upload a .txt
# file renamed to .xlsx).
import os

# 'Path' from pathlib is a more modern, readable way to work with file paths
# than plain strings. We use it alongside 'os' for clarity.
from pathlib import Path

# Type hints: these don't change how the code runs, but they tell any human
# (or IDE) reading the code exactly what type of data goes in and out of
# each function. This is a big part of "beginner friendly" code -- you can
# see what a function expects without reading its entire body.
from typing import Union, List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# FUNCTION 1: list_sheet_names
# ---------------------------------------------------------------------------
def list_sheet_names(file_path_or_buffer: Union[str, Any]) -> List[str]:
    """
    Look inside an Excel workbook and return the names of every sheet it
    contains, WITHOUT loading all the actual row data into memory.

    Why do we need this at all?
        Excel workbooks can contain multiple tabs/sheets (e.g. "Jan2024",
        "Feb2024", "Summary"). Before we load data, we may want to show
        the user a dropdown of sheet names so they can pick the correct one.
        Using pandas.ExcelFile() lets us peek at sheet names cheaply,
        instead of loading every sheet's full data just to see their names.

    Parameters
    ----------
    file_path_or_buffer : str OR a file-like object
        This can be either:
            - a plain string file path, e.g. "dataset/procurement.xlsx"
            - an in-memory file object, such as what Streamlit's
              st.file_uploader() returns when a user uploads a file in
              the browser (it behaves like a file, but isn't one on disk).

    Returns
    -------
    List[str]
        A list of sheet name strings, e.g. ["Sheet1", "Q1_Data", "Q2_Data"]

    Raises
    ------
    ValueError
        If the file cannot be opened as an Excel workbook at all
        (e.g. it's corrupted, or not really an Excel file).
    """

    # We wrap this in a try/except because opening a broken or non-Excel
    # file will throw a low-level error from the underlying openpyxl
    # library. We catch that and re-raise it as a clearer, friendlier
    # error message that a beginner (or the Streamlit UI) can display.
    try:
        # pandas.ExcelFile() opens the workbook "lazily" -- it reads the
        # workbook's internal structure (like a table of contents) but
        # does NOT load every sheet's data into memory yet. This is fast
        # and cheap, which is exactly what we want just to list sheet names.
        excel_file = pd.ExcelFile(file_path_or_buffer, engine="openpyxl")

        # .sheet_names is an attribute (not a function) that pandas fills
        # in automatically once the file is opened. It's already a list
        # of strings, so we can return it directly.
        return excel_file.sheet_names

    except Exception as error:
        # 'as error' captures whatever the original Python error was so we
        # can include its message in our own, more descriptive error.
        # Raising a new error here (instead of returning None or an empty
        # list) is intentional: a broken file should stop the pipeline
        # loudly and immediately, not fail silently three steps later.
        raise ValueError(
            f"Could not read sheet names from the Excel file. "
            f"It may be corrupted or not a valid .xlsx file. "
            f"Original error: {error}"
        )


# ---------------------------------------------------------------------------
# FUNCTION 2: validate_file_exists
# ---------------------------------------------------------------------------
def validate_file_exists(file_path: str) -> bool:
    """
    Confirm that a given file path actually points to a real file on disk,
    and that it has an Excel-compatible file extension.

    NOTE: This check is only meaningful for plain file paths (e.g. when
    testing this script from the command line, or reading from the
    'dataset/' folder). It is skipped for in-memory Streamlit uploads,
    since those don't have a real path on disk -- Streamlit already
    guarantees the object exists if the user picked a file.

    Parameters
    ----------
    file_path : str
        A path to a file, e.g. "dataset/procurement_2024.xlsx"

    Returns
    -------
    bool
        True if the file exists AND ends in .xlsx or .xls

    Raises
    ------
    FileNotFoundError
        If the path does not point to an existing file.
    ValueError
        If the file exists but is not an Excel file (wrong extension).
    """

    # Path(file_path) converts the plain string into a Path object, which
    # gives us convenient methods like .exists() and .suffix below.
    path_object = Path(file_path)

    # .exists() checks the actual filesystem to see if something is really
    # sitting at that location. This returns False for typos, wrong
    # folders, or files that were never uploaded/saved.
    if not path_object.exists():
        # We raise immediately rather than returning False, because a
        # missing file is a "stop everything" situation -- there is no
        # sensible way to continue the pipeline without it.
        raise FileNotFoundError(f"No file found at path: {file_path}")

    # .suffix returns the file extension including the dot, e.g. ".xlsx".
    # .lower() makes the check case-insensitive (so ".XLSX" still passes).
    file_extension = path_object.suffix.lower()

    # We only accept modern (.xlsx) and legacy (.xls) Excel formats.
    if file_extension not in [".xlsx", ".xls"]:
        raise ValueError(
            f"File '{file_path}' does not appear to be an Excel file "
            f"(found extension '{file_extension}'). Expected .xlsx or .xls."
        )

    # If we reach this line, both checks passed: the file exists AND it
    # has a valid Excel extension.
    return True


# ---------------------------------------------------------------------------
# FUNCTION 3: load_excel_file
# ---------------------------------------------------------------------------
def load_excel_file(
    file_path_or_buffer: Union[str, Any],
    sheet_name: Union[str, int] = 0
) -> pd.DataFrame:
    """
    Load a single sheet of an Excel workbook into a pandas DataFrame.

    This is the main "front door" function of the whole application --
    every other module downstream (cleanData.py, analyzeData.py, etc.)
    ultimately receives the DataFrame that this function produces.

    Parameters
    ----------
    file_path_or_buffer : str OR file-like object
        A plain file path (for local/CLI testing) OR the in-memory file
        object handed to us by Streamlit's st.file_uploader().
    sheet_name : str or int, default 0
        Which sheet to load.
            - An int (like 0) loads the sheet by POSITION (0 = first sheet).
            - A str (like "Q1_Data") loads the sheet by NAME.
        Defaulting to 0 means: "if the caller doesn't specify, just load
        the first sheet in the workbook" -- a sensible default for most
        procurement exports, which usually have a single relevant sheet.

    Returns
    -------
    pd.DataFrame
        The loaded spreadsheet data as a pandas DataFrame, completely
        untouched/unmodified (no cleaning happens here on purpose).

    Raises
    ------
    ValueError
        If the file is unreadable, the sheet doesn't exist, or the
        resulting DataFrame is empty.
    """

    # Step A: If we were given a plain string path (not an uploaded file
    # object), run our filesystem checks from validate_file_exists() first.
    # We check "isinstance(..., str)" because Streamlit's uploaded file
    # object is NOT a string -- it's a special buffer object -- so this
    # condition naturally skips the disk check for uploads.
    if isinstance(file_path_or_buffer, str):
        validate_file_exists(file_path_or_buffer)

    # Step B: Actually read the Excel sheet into a DataFrame.
    try:
        dataframe = pd.read_excel(
            file_path_or_buffer,   # the file path or uploaded buffer
            sheet_name=sheet_name,  # which sheet to load (name or index)
            engine="openpyxl"      # openpyxl is the engine that reads .xlsx
        )
    except Exception as error:
        # Any failure here (bad sheet name, corrupted file, permissions
        # issue, etc.) gets converted into one clear, friendly message.
        raise ValueError(
            f"Failed to load Excel data from sheet '{sheet_name}'. "
            f"Original error: {error}"
        )

    # Step C: Make sure we didn't just load an empty sheet. An empty
    # DataFrame downstream would cause confusing errors much later in the
    # pipeline (e.g. "no numeric columns found"), so we catch it here,
    # at the source, with a clear message.
    if dataframe.empty:
        raise ValueError(
            f"The sheet '{sheet_name}' was loaded successfully, but it "
            f"contains no data (0 rows)."
        )

    # Step D: Return the raw, untouched DataFrame. Cleaning is intentionally
    # NOT done here -- that is the single responsibility of cleanData.py.
    return dataframe


# ---------------------------------------------------------------------------
# FUNCTION 4: validate_required_columns
# ---------------------------------------------------------------------------
def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> bool:
    """
    Check whether a DataFrame contains a specific set of expected columns.

    IMPORTANT DESIGN NOTE (why 'required_columns' is optional):
    Project Walter is designed to AUTO-DETECT which columns represent
    things like "cost", "vendor", or "date" later on, in analyzeData.py --
    because every hospital department's Excel export names its columns
    differently (e.g. "Total Cost" vs "Amount" vs "Spend_USD"). Because of
    this, data.py cannot always know in advance the exact column names to
    require.

    So this function supports two use cases:
        1. General validation: pass in a list of columns YOU know must be
           present for a specific dataset (e.g. during testing, or if your
           department always exports a column literally named "PO Number").
        2. Skip validation: pass nothing (None), and this function just
           confirms the DataFrame has at least one column and is not empty,
           deferring smart "does this column mean cost?" logic to
           analyzeData.py.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    required_columns : list of str, optional
        Column names that MUST be present. If None, only a basic
        "does this DataFrame have any columns at all" check is performed.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If the DataFrame has no columns, or if any required column is
        missing.
    """

    # Basic sanity check: a DataFrame with zero columns is unusable no
    # matter what, so we check this regardless of whether specific
    # required columns were given.
    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns at all -- cannot proceed.")

    # If the caller didn't specify particular required columns, we've
    # already done the only check that makes sense at this stage, so we
    # can return True here.
    if required_columns is None:
        return True

    # df.columns gives us all column names currently in the DataFrame.
    # Converting to a set makes the "is this missing?" check below fast
    # and easy to read.
    existing_columns = set(df.columns)

    # A Python list comprehension: build a list of every column name from
    # 'required_columns' that is NOT found in 'existing_columns'.
    missing_columns = [
        col for col in required_columns if col not in existing_columns
    ]

    # If the missing_columns list has anything in it, at least one
    # required column was not found -- so we raise a clear error that
    # names exactly which columns are missing (very helpful for debugging).
    if missing_columns:
        raise ValueError(
            f"The dataset is missing required column(s): {missing_columns}. "
            f"Columns found in file: {list(df.columns)}"
        )

    # All required columns were found.
    return True


# ---------------------------------------------------------------------------
# FUNCTION 5: get_basic_file_info
# ---------------------------------------------------------------------------
def get_basic_file_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Collect a small "fact sheet" describing the loaded dataset.

    This function does not print anything -- it just returns a plain
    Python dictionary of facts. Keeping "collecting the facts" separate
    from "displaying the facts" (done in display_data_summary below) means
    main.py's Streamlit UI can later use this exact same dictionary to
    render nice metric boxes on screen, instead of only being able to
    print plain text to a console.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded (and NOT yet cleaned) dataset.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys:
            "num_rows"        -> int, total number of rows
            "num_columns"     -> int, total number of columns
            "column_names"    -> list of str, every column's name
            "data_types"      -> dict, column name -> data type as string
            "missing_values"  -> dict, column name -> count of empty cells
            "first_five_rows" -> a small DataFrame preview (df.head())
    """

    # df.shape returns a tuple like (num_rows, num_columns). We "unpack"
    # it into two separate variables in one line.
    num_rows, num_columns = df.shape

    # df.columns is a special pandas Index object; wrapping it in list()
    # converts it into a plain Python list of strings, which is easier
    # to display and to pass around to other functions/UI elements.
    column_names = list(df.columns)

    # df.dtypes tells pandas' internal data type for every column (e.g.
    # int64, float64, object [=text], datetime64). We convert each type
    # to a plain string (str(dtype)) so the dictionary is simple and
    # human-readable rather than full of pandas-specific objects.
    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # df.isnull() returns a same-shaped DataFrame of True/False values
    # (True = the cell is empty/missing). .sum() then adds up the Trues
    # per column (True counts as 1, False as 0), giving us a count of
    # missing values for every column.
    missing_values = df.isnull().sum().to_dict()

    # df.head() returns the first 5 rows by default -- exactly what was
    # requested. We keep this as an actual small DataFrame (not text) so
    # the Streamlit UI can render it as a nice interactive table later.
    first_five_rows = df.head(5)

    # Package everything into one dictionary and return it.
    return {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "column_names": column_names,
        "data_types": data_types,
        "missing_values": missing_values,
        "first_five_rows": first_five_rows,
    }


# ---------------------------------------------------------------------------
# FUNCTION 6: display_data_summary
# ---------------------------------------------------------------------------
def display_data_summary(df: pd.DataFrame) -> None:
    """
    Print a clean, human-readable summary of the dataset to the console.

    This function exists mainly for TESTING this file on its own (see the
    "if __name__ == '__main__':" block at the bottom of this file) and for
    quick command-line debugging. The Streamlit UI in main.py will instead
    call get_basic_file_info() directly and render the results as proper
    on-screen widgets -- but the underlying facts are identical either way.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset to summarize.

    Returns
    -------
    None
        This function only prints to the console; it doesn't return
        anything, because printing is its entire purpose.
    """

    # Reuse get_basic_file_info() so we never calculate these facts twice
    # in two different places (avoids the two versions ever disagreeing).
    info = get_basic_file_info(df)

    # A simple divider line makes console output easier to read.
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    # --- Number of rows and columns ---
    print(f"Number of rows:    {info['num_rows']}")
    print(f"Number of columns: {info['num_columns']}")

    # --- Column names ---
    print("\nColumn names:")
    # enumerate() gives us both a running count (i) and each column name,
    # starting the count at 1 so it reads naturally to a human (not 0).
    for i, col in enumerate(info["column_names"], start=1):
        print(f"  {i}. {col}")

    # --- Data types ---
    print("\nData types:")
    # .items() lets us loop over a dictionary's (key, value) pairs at once.
    for col, dtype in info["data_types"].items():
        print(f"  {col}: {dtype}")

    # --- Missing values ---
    print("\nMissing values per column:")
    for col, missing_count in info["missing_values"].items():
        print(f"  {col}: {missing_count} missing")

    # --- First five rows ---
    print("\nFirst 5 rows of data:")
    # Printing a DataFrame directly uses pandas' own nicely aligned table
    # formatting, which is easy to read in a plain console.
    print(info["first_five_rows"])

    print("=" * 60)


# ---------------------------------------------------------------------------
# STANDALONE TEST BLOCK
# ---------------------------------------------------------------------------
# Everything below only runs if you execute this file directly, e.g.:
#     python src/data.py
# It will NOT run if this file is imported by another file (like main.py).
# This lets us test data.py completely on its own, before any other module
# in the pipeline even exists.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # We build a tiny sample Excel file on the fly so this test block can
    # run immediately, without requiring you to already have a real
    # procurement spreadsheet in the dataset/ folder.
    sample_data = {
        "PO Number": ["PO-1001", "PO-1002", "PO-1003", "PO-1004", "PO-1005", "PO-1006"],
        "Vendor": ["Medtronic", "GE Healthcare", "Stryker", "Medtronic", "Philips", None],
        "Item Description": [
            "Infusion Pump", "Patient Monitor", "Surgical Drill",
            "Infusion Pump", "Ultrasound Probe", "Ventilator Filter"
        ],
        "Total Cost": [4500.00, 12800.50, 3200.00, 4600.00, 980.25, 150.00],
        "Order Date": pd.to_datetime([
            "2024-01-05", "2024-01-12", "2024-02-01",
            "2024-02-10", "2024-03-03", "2024-03-15"
        ]),
    }
    sample_df = pd.DataFrame(sample_data)

    # Save this sample DataFrame to an actual .xlsx file inside dataset/,
    # so we are testing the REAL load_excel_file() function end-to-end
    # (reading from disk), not just working with an in-memory DataFrame.
    test_file_path = "dataset/sample_procurement_data.xlsx"
    sample_df.to_excel(test_file_path, index=False, engine="openpyxl")
    print(f"Created a sample test file at: {test_file_path}\n")

    # 1. Test listing sheet names.
    sheet_names = list_sheet_names(test_file_path)
    print(f"Sheets found in workbook: {sheet_names}\n")

    # 2. Test loading the file into a DataFrame.
    loaded_df = load_excel_file(test_file_path, sheet_name=sheet_names[0])
    print("File loaded successfully.\n")

    # 3. Test required-column validation with a realistic required list.
    validate_required_columns(loaded_df, required_columns=["PO Number", "Total Cost"])
    print("Required columns check passed: 'PO Number' and 'Total Cost' both exist.\n")

    # 4. Display the full data summary (rows, columns, dtypes, missing, head).
    display_data_summary(loaded_df)

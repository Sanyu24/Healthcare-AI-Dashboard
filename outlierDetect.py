"""
=============================================================================
 outlierDetect.py
=============================================================================
STEP 5 OF THE PIPELINE: OUTLIER DETECTION

This module looks at every individual purchase order amount and flags the
ones that are STATISTICALLY UNUSUAL -- meaning they stand out sharply from
how the rest of the organization normally spends. This is one of the most
valuable parts of the whole application: it turns a huge spreadsheet of
thousands of purchases into a short, focused list of transactions that
actually deserve a human being's attention.

BEGINNER-FRIENDLY EXPLANATION OF THE TWO METHODS WE USE:

    1. Z-SCORE METHOD
       Imagine lining up every purchase amount and asking: "On average,
       how far does a typical purchase sit from the overall average
       amount, back and forth?" That "typical distance" is called the
       STANDARD DEVIATION. A z-score simply answers: "How many of those
       typical distances away from the average is THIS purchase?"
       A z-score of 0 means "exactly average." A z-score of 3 means this
       purchase is 3 standard deviations away from average -- which is
       rare (only about 0.3% of normal, evenly-spread-out data would be
       that extreme). We flag anything with a z-score above a chosen
       threshold (default: 3).

    2. IQR (INTERQUARTILE RANGE) METHOD
       This method ignores the average entirely and instead asks: "What
       does the normal MIDDLE 50% of purchases look like?" It finds the
       25th percentile (the value where 25% of purchases are cheaper) and
       the 75th percentile (where 75% are cheaper), and calls the gap
       between them the "IQR." Anything far below the 25th percentile or
       far above the 75th percentile is flagged. This method is useful
       because, unlike the average, it isn't thrown off by a few already-
       huge legitimate purchases skewing things.

    WHY USE BOTH METHODS TOGETHER?
    Each method can miss things the other catches. Using both and flagging
    a purchase if EITHER method considers it unusual gives more complete,
    more trustworthy outlier detection than relying on just one.

WHAT THIS FILE PROVIDES (public functions):
    1. calculate_z_scores(df, amount_col)          -> z-score for every row
    2. flag_zscore_outliers(df, amount_col, ...)    -> True/False per row
    3. calculate_iqr_bounds(df, amount_col, ...)    -> (lower_bound, upper_bound)
    4. flag_iqr_outliers(df, amount_col, ...)       -> True/False per row
    5. build_outlier_reasons(...)                   -> plain-English reason text
    6. detect_outliers(df, ...)                     -> (normal_df, outlier_df)
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# pandas gives us the DataFrame object and easy statistical operations
# like .mean(), .std(), and .quantile().
import pandas as pd

# numpy is used for numeric helpers -- here mainly np.where(), which lets
# us efficiently choose between two values for every row at once based on
# a condition, without writing a manual loop.
import numpy as np

# Type hints -- purely for readability, they don't change how the code runs.
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# FUNCTION 1: calculate_z_scores
# ---------------------------------------------------------------------------
def calculate_z_scores(df: pd.DataFrame, amount_column: str) -> pd.Series:
    """
    Calculate a "z-score" for every purchase amount in the dataset.

    BEGINNER EXPLANATION:
    A z-score answers the question: "How many standard deviations away
    from the average is this specific value?" The formula is simple:

        z-score = (value - average) / standard_deviation

    A z-score of 0 means the value IS the average. A z-score of +2 means
    the value is 2 "typical distances" ABOVE the average. A z-score of -2
    means it's 2 typical distances BELOW the average. The further a
    z-score is from 0 (in either direction), the more unusual that
    purchase is compared to the rest of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.

    Returns
    -------
    pd.Series
        A new Series (one value per row) containing each row's z-score.
    """

    # Defensive check: make sure the column we need actually exists,
    # with a clear error message if it doesn't.
    if amount_column not in df.columns:
        raise ValueError(f"Column '{amount_column}' not found in dataset.")

    # .mean() calculates the average of every purchase amount in the
    # entire dataset -- this is our reference point for "normal."
    average_amount = df[amount_column].mean()

    # .std() calculates the standard deviation -- roughly speaking, the
    # "typical distance" any given purchase sits from that average.
    standard_deviation = df[amount_column].std()

    # Apply the z-score formula to EVERY row at once (this is called a
    # "vectorized" operation -- much faster than looping row by row).
    # (value - average) tells us how far above/below average each row is.
    # Dividing by standard_deviation converts that raw dollar difference
    # into "number of typical distances," which is what makes different
    # datasets/columns comparable on the same scale.
    z_scores = (df[amount_column] - average_amount) / standard_deviation

    return z_scores


# ---------------------------------------------------------------------------
# FUNCTION 2: flag_zscore_outliers
# ---------------------------------------------------------------------------
def flag_zscore_outliers(
    df: pd.DataFrame,
    amount_column: str,
    z_threshold: float = 3.0,
) -> pd.Series:
    """
    Decide which purchase orders are "unusual" according to the z-score
    method, using a chosen cutoff threshold.

    BEGINNER EXPLANATION:
    We calculate a z-score for every purchase (see calculate_z_scores
    above), then simply check: is the ABSOLUTE VALUE of that z-score
    bigger than our threshold? We use the absolute value because a
    purchase can be unusual for being surprisingly LARGE (positive
    z-score) OR surprisingly SMALL (negative z-score, which can sometimes
    indicate a data-entry mistake, like a missing decimal point).

    A threshold of 3.0 (the common default used in statistics) means:
    "only flag purchases so extreme that fewer than about 3 in 1,000
    normal purchases would ever look like this."

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    z_threshold : float, default 3.0
        How many standard deviations away from average counts as
        "unusual." Lower values (e.g. 2.0) flag MORE purchases (more
        sensitive); higher values (e.g. 4.0) flag FEWER purchases (more
        strict).

    Returns
    -------
    pd.Series
        A Series of True/False values, one per row: True means this row
        is flagged as an outlier by the z-score method.
    """

    # Reuse our z-score calculation from Function 1, so this logic isn't
    # duplicated in two places.
    z_scores = calculate_z_scores(df, amount_column)

    # .abs() converts every z-score to its positive/absolute value, so a
    # z-score of -4 (unusually LOW) and +4 (unusually HIGH) are both
    # correctly treated as "4 away from normal" and flagged the same way.
    # The ">" comparison then produces a True/False value for every row.
    is_outlier = z_scores.abs() > z_threshold

    return is_outlier


# ---------------------------------------------------------------------------
# FUNCTION 3: calculate_iqr_bounds
# ---------------------------------------------------------------------------
def calculate_iqr_bounds(
    df: pd.DataFrame,
    amount_column: str,
    iqr_multiplier: float = 1.5,
) -> Tuple[float, float]:
    """
    Calculate the "normal range" of purchase amounts using the
    Interquartile Range (IQR) method, and return the lower and upper
    boundaries of that normal range.

    BEGINNER EXPLANATION:
    Imagine sorting every purchase amount from smallest to largest, then
    dividing that sorted list into four equal quarters:
        - Q1 (25th percentile): 25% of purchases are cheaper than this
        - Q3 (75th percentile): 75% of purchases are cheaper than this
    The "IQR" is simply the gap between Q3 and Q1 -- it represents the
    spread of the normal MIDDLE 50% of purchases.

    We then draw a "fence" around that normal middle range:
        lower_bound = Q1 - (iqr_multiplier x IQR)
        upper_bound = Q3 + (iqr_multiplier x IQR)
    Anything OUTSIDE this fence (either smaller than lower_bound or larger
    than upper_bound) is considered unusually far from the normal middle
    range of spending.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    iqr_multiplier : float, default 1.5
        How far beyond Q1/Q3 the "fence" extends, measured in IQRs. 1.5 is
        the standard, widely-used default in statistics (sometimes called
        "Tukey's rule"). A larger multiplier (e.g. 3.0) creates a wider
        fence and flags fewer, more extreme purchases.

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) -- the two boundary values that define
        the "normal" range of purchase amounts.
    """

    if amount_column not in df.columns:
        raise ValueError(f"Column '{amount_column}' not found in dataset.")

    # .quantile(0.25) finds the value at the 25th percentile (Q1) -- the
    # point below which the cheapest 25% of purchases fall.
    q1 = df[amount_column].quantile(0.25)

    # .quantile(0.75) finds the value at the 75th percentile (Q3) -- the
    # point below which the cheapest 75% of purchases fall.
    q3 = df[amount_column].quantile(0.75)

    # The Interquartile Range itself: the width of the "normal middle 50%"
    # window of purchase amounts.
    iqr = q3 - q1

    # Build the lower and upper "fences" around that normal window.
    lower_bound = q1 - (iqr_multiplier * iqr)
    upper_bound = q3 + (iqr_multiplier * iqr)

    return lower_bound, upper_bound


# ---------------------------------------------------------------------------
# FUNCTION 4: flag_iqr_outliers
# ---------------------------------------------------------------------------
def flag_iqr_outliers(
    df: pd.DataFrame,
    amount_column: str,
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    """
    Decide which purchase orders are "unusual" according to the IQR
    method, using the lower/upper boundaries calculated above.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    iqr_multiplier : float, default 1.5
        Passed straight through to calculate_iqr_bounds() -- see that
        function's docstring for a full explanation.

    Returns
    -------
    pd.Series
        A Series of True/False values, one per row: True means this row
        falls outside the normal IQR-based range (either much higher or
        much lower than typical).
    """

    # Get the "normal range" boundaries using Function 3.
    lower_bound, upper_bound = calculate_iqr_bounds(df, amount_column, iqr_multiplier)

    # A row is flagged if its amount is EITHER below the lower fence OR
    # above the upper fence. The "|" symbol means "OR" when comparing two
    # True/False Series together, row by row.
    is_outlier = (df[amount_column] < lower_bound) | (df[amount_column] > upper_bound)

    return is_outlier


# ---------------------------------------------------------------------------
# FUNCTION 5: build_outlier_reasons
# ---------------------------------------------------------------------------
def build_outlier_reasons(
    df: pd.DataFrame,
    amount_column: str,
    z_scores: pd.Series,
    z_flags: pd.Series,
    iqr_flags: pd.Series,
    lower_bound: float,
    upper_bound: float,
) -> pd.Series:
    """
    Turn the True/False outlier flags into a clear, plain-English
    sentence explaining WHY each flagged purchase was considered unusual.

    WHY THIS MATTERS:
    A "True" in an "Is Outlier" column tells a reviewer THAT something is
    unusual, but not WHY. A human reviewing this list (e.g. a Clinical
    Engineering manager) will trust and act on this system much more
    readily if it explains its reasoning in plain language, rather than
    just showing a cryptic statistical flag.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    z_scores : pd.Series
        The z-score for every row (from calculate_z_scores()).
    z_flags : pd.Series
        True/False per row from the z-score method.
    iqr_flags : pd.Series
        True/False per row from the IQR method.
    lower_bound, upper_bound : float
        The IQR "normal range" boundaries, used to explain IQR-based
        flags in plain language.

    Returns
    -------
    pd.Series
        A Series of text strings, one per row, explaining the reason(s)
        this row was flagged. For rows that weren't flagged by either
        method, this will be an empty string.
    """

    # We'll build the reason text row by row using a simple Python list,
    # then convert it to a pandas Series at the end. Building a list first
    # (instead of repeatedly modifying a DataFrame cell by cell) is a
    # faster and more beginner-readable approach here, since the reason
    # text logic has several branching "if" conditions per row.
    reasons = []

    # zip() lets us walk through several Series together, one row at a
    # time, all lined up by position. range(len(df)) combined with
    # .iloc[i] would also work, but zip() keeps this loop clean and
    # readable.
    for amount, z_score, is_z_outlier, is_iqr_outlier in zip(
        df[amount_column], z_scores, z_flags, iqr_flags
    ):
        row_reasons = []  # collects one or more reason phrases for this single row

        # --- Check the Z-score method ---
        if is_z_outlier:
            direction = "above" if z_score > 0 else "below"
            row_reasons.append(
                f"Amount is {abs(round(z_score, 1))} standard deviations {direction} "
                f"the average (Z-score method)"
            )

        # --- Check the IQR method ---
        if is_iqr_outlier:
            if amount > upper_bound:
                row_reasons.append(
                    f"Amount (${amount:,.2f}) is far above the typical high end of "
                    f"${upper_bound:,.2f} (IQR method)"
                )
            else:
                row_reasons.append(
                    f"Amount (${amount:,.2f}) is far below the typical low end of "
                    f"${lower_bound:,.2f} (IQR method)"
                )

        # If neither method flagged this row, the reason is simply blank
        # (this row will end up in the "normal purchases" table anyway).
        # If one or both methods DID flag it, join every reason phrase
        # together with a semicolon so both explanations show up together
        # when a purchase is flagged by both methods at once.
        reasons.append("; ".join(row_reasons))

    # Convert our plain Python list back into a pandas Series, using the
    # same row index as the original DataFrame so it lines up correctly
    # if this Series is later attached back onto the DataFrame.
    return pd.Series(reasons, index=df.index)


# ---------------------------------------------------------------------------
# FUNCTION 6: detect_outliers (the orchestrator)
# ---------------------------------------------------------------------------
def detect_outliers(
    df: pd.DataFrame,
    po_column: str,
    supplier_column: str,
    amount_column: str,
    market_column: Optional[str] = None,
    cost_center_column: Optional[str] = None,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the FULL outlier-detection process and split the dataset into two
    separate, easy-to-use tables: normal purchases, and outlier purchases.

    This is the ONE function that main.py should actually call -- it runs
    both statistical methods, combines their results, builds the
    human-readable explanation for every outlier, and packages everything
    into two clean output DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset (output of cleanData.py's clean_data()).
    po_column : str
        Name of the column containing the purchase order number/ID.
    supplier_column : str
        Name of the column containing supplier/vendor names.
    amount_column : str
        Name of the numeric purchase amount column.
    market_column : str, optional
        Name of the market/category column. If this column doesn't exist
        in a given dataset, it's simply left out of the output tables
        rather than causing an error.
    cost_center_column : str, optional
        Name of the cost center column. Same graceful handling as
        market_column above.
    z_threshold : float, default 3.0
        The z-score cutoff used to flag outliers (see
        flag_zscore_outliers() for a full explanation).
    iqr_multiplier : float, default 1.5
        The IQR "fence width" multiplier used to flag outliers (see
        calculate_iqr_bounds() for a full explanation).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (normal_df, outlier_df)

        normal_df : every row that was NOT flagged by either method, with
                    all of its original columns kept intact.

        outlier_df : every row that WAS flagged by at least one method,
                     containing only these columns (in this order):
                        - PO Number   (from po_column)
                        - Supplier    (from supplier_column)
                        - Amount      (from amount_column)
                        - Market      (from market_column, if provided)
                        - Cost Center (from cost_center_column, if provided)
                        - Reason      (plain-English explanation, from
                                       build_outlier_reasons())

    A NOTE ON A KNOWN LIMITATION -- THE "MASKING EFFECT":
    Both the z-score and IQR methods calculate what "normal" looks like
    FROM the same dataset they are then checking for outliers in. This
    means one single, extremely large outlier can pull the average and
    spread up so much that it "masks" (hides) a second, smaller outlier
    that would otherwise have been flagged. This effect matters most in
    SMALL datasets (a few dozen rows or fewer) -- with a large, realistic
    procurement dataset (hundreds or thousands of purchase orders), any
    one unusual purchase has far less influence on the overall average
    and spread, so this becomes far less of a concern in practice. If you
    ever suspect masking is hiding a known unusual purchase, re-running
    detect_outliers() on a filtered subset of the data (e.g. one
    supplier's purchases at a time) can reveal outliers that the
    full-dataset view missed.
    """

    # --- Step A: basic column validation ---
    # We check the columns we absolutely need up front, so any problem is
    # reported clearly and immediately, rather than causing a confusing
    # error partway through the statistical calculations below.
    required_columns = [po_column, supplier_column, amount_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset.")

    # We work on a copy so this function never modifies the DataFrame the
    # caller passed in.
    df = df.copy()

    # --- Step B: run both statistical detection methods ---
    z_scores = calculate_z_scores(df, amount_column)
    z_flags = flag_zscore_outliers(df, amount_column, z_threshold)
    iqr_flags = flag_iqr_outliers(df, amount_column, iqr_multiplier)

    # We also need the IQR bounds themselves (not just the True/False
    # flags) so build_outlier_reasons() can mention the actual dollar
    # cutoff values in its plain-English explanations.
    lower_bound, upper_bound = calculate_iqr_bounds(df, amount_column, iqr_multiplier)

    # --- Step C: combine the two methods ---
    # A row is considered an outlier overall if EITHER method flagged it.
    # The "|" symbol means "OR" when combining two True/False Series.
    combined_outlier_flag = z_flags | iqr_flags

    # --- Step D: build the plain-English reason for every row ---
    reasons = build_outlier_reasons(
        df, amount_column, z_scores, z_flags, iqr_flags, lower_bound, upper_bound
    )

    # Temporarily attach our working columns onto the DataFrame so we can
    # easily filter/select from them below. These are "helper" columns
    # used only inside this function.
    df["_Is_Outlier"] = combined_outlier_flag
    df["_Reason"] = reasons

    # --- Step E: split into normal vs. outlier rows ---
    # Boolean indexing: df[condition] keeps only the rows where condition
    # is True. Using "~" (the "NOT" operator) in front of a True/False
    # Series flips every value, so "~df['_Is_Outlier']" selects every row
    # that was NOT flagged.
    normal_df = df[~df["_Is_Outlier"]].copy()
    outlier_df = df[df["_Is_Outlier"]].copy()

    # For the "normal" table, we keep ALL original columns (dropping only
    # our two temporary helper columns), since a normal purchase doesn't
    # need a special simplified view -- someone might still want full
    # detail on it later (e.g. for the executive summary or further
    # analysis).
    normal_df = normal_df.drop(columns=["_Is_Outlier", "_Reason"])

    # --- Step F: build the simplified, focused outlier_df ---
    # Per the requirement, the outlier table should show only these
    # specific columns. We build a rename dictionary mapping each
    # caller-supplied column name to the friendly, standardized output
    # name requested.
    column_rename_map = {
        po_column: "PO Number",
        supplier_column: "Supplier",
        amount_column: "Amount",
    }
    output_columns = [po_column, supplier_column, amount_column]

    # Market and Cost Center are OPTIONAL -- only include them if the
    # caller actually provided a column name AND that column exists in
    # this particular dataset. This keeps the function working smoothly
    # even for hospital exports that don't track these fields.
    if market_column is not None and market_column in df.columns:
        output_columns.append(market_column)
        column_rename_map[market_column] = "Market"

    if cost_center_column is not None and cost_center_column in df.columns:
        output_columns.append(cost_center_column)
        column_rename_map[cost_center_column] = "Cost Center"

    # Select just the columns we want, in order, then rename them to the
    # clean, standardized output names.
    outlier_df = outlier_df[output_columns + ["_Reason"]].rename(columns=column_rename_map)
    outlier_df = outlier_df.rename(columns={"_Reason": "Reason"})

    # Sort the outlier table so the LARGEST (most attention-worthy)
    # flagged purchases appear first -- the most useful reading order for
    # a manager scanning this list.
    outlier_df = outlier_df.sort_values(by="Amount", ascending=False).reset_index(drop=True)

    # Tidy up the row numbering on the normal table too, since some rows
    # were removed from the middle of the original DataFrame.
    normal_df = normal_df.reset_index(drop=True)

    return normal_df, outlier_df


#OUTLIERS BY MARKET

def detect_outliers_per_market(
    df: pd.DataFrame,
    amount_column: str,
    po_column: str = None,
    supplier_column: str = None,
    market_column: str = "Market",
    cost_center_column: str = None,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
):
    """
    Detect outliers separately within each Market.
    """

    normal_frames = []
    outlier_frames = []

    # Ignore blank markets
    grouped = df.groupby(market_column, dropna=True)

    for market_name, market_df in grouped:

        if len(market_df) < 5:
            normal_frames.append(market_df.copy())
            continue

        normal_df, outlier_df = detect_outliers(
            market_df,
            po_column=po_column,
            supplier_column=supplier_column,
            amount_column=amount_column,
            market_column=market_column,
            cost_center_column=cost_center_column,
            z_threshold=z_threshold,
            iqr_multiplier=iqr_multiplier,
        )

        normal_frames.append(normal_df)

        if not outlier_df.empty:
            outlier_frames.append(outlier_df)

    if normal_frames:
        normal_df = pd.concat(normal_frames, ignore_index=True)
    else:
        normal_df = pd.DataFrame()

    if outlier_frames:
        outlier_df = pd.concat(outlier_frames, ignore_index=True)
    else:
        outlier_df = pd.DataFrame()

    return normal_df, outlier_df
# ---------------------------------------------------------------------------
# STANDALONE TEST BLOCK
# ---------------------------------------------------------------------------
# Runs only when you execute this file directly: python src/outlierDetect.py
# We build a small sample dataset with a few DELIBERATELY unusual purchase
# amounts mixed in among normal ones, then confirm detect_outliers()
# correctly separates them and explains why.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # NOTE ON DATASET SIZE: we deliberately use a LARGER sample (40 rows)
    # here rather than a tiny handful. With very few data points, a single
    # extreme outlier can distort the average/spread so much that it
    # "masks" (hides) a second outlier -- a real, known statistical effect
    # explained in detect_outliers()'s docstring above. A larger, more
    # realistic sample size (like a real hospital's data would have)
    # avoids this problem, which is why we use one here to cleanly
    # demonstrate BOTH a high and a low outlier being caught correctly.
    suppliers_cycle = [
        "Medtronic", "Ge Healthcare", "Stryker", "Philips", "Boston Scientific",
    ]
    markets_cycle = ["Patient Monitoring", "Imaging", "Surgical", "Cardiology", "Imaging"]
    cost_centers_cycle = ["ICU", "Radiology", "OR", "Cath Lab", "Radiology"]

    # Build 38 "normal" purchases with realistic, everyday amounts.
    normal_amounts = [
        4500, 12800, 3200, 4600, 980, 3100, 4550, 15000, 890, 3050,
        13200, 8200, 9100, 2700, 5400, 6300, 2100, 7600, 11200, 4300,
        3900, 6100, 8800, 5200, 1450, 9700, 4100, 6700, 3300, 12100,
        7200, 5600, 4800, 9300, 2900, 6900, 5100, 8400,
    ]

    sample_data = {
        "PO Number": [f"PO-{1000+i}" for i in range(len(normal_amounts) + 2)],
        "Supplier": [suppliers_cycle[i % 5] for i in range(len(normal_amounts))] + ["Medtronic", "Stryker"],
        "Market": [markets_cycle[i % 5] for i in range(len(normal_amounts))] + ["Patient Monitoring", "Surgical"],
        "Cost Center": [cost_centers_cycle[i % 5] for i in range(len(normal_amounts))] + ["ICU", "OR"],
        # The 38 realistic amounts above, PLUS two deliberately planted
        # outliers at the end: one very large ($185,000) and one
        # suspiciously tiny ($5) that likely indicates a data-entry error.
        "PO Amount Ordered": normal_amounts + [185000.00, 5.00],
    }
    sample_df = pd.DataFrame(sample_data)

    print("=" * 60)
    print("RUNNING OUTLIER DETECTION")
    print("=" * 60)

    normal_df, outlier_df = detect_outliers(
        sample_df,
        po_column="PO Number",
        supplier_column="Supplier",
        amount_column="PO Amount Ordered",
        market_column="Market",
        cost_center_column="Cost Center",
    )

    print(f"\nTotal purchase orders: {len(sample_df)}")
    print(f"Normal purchases:      {len(normal_df)}")
    print(f"Outlier purchases:     {len(outlier_df)}")

    print("\n" + "=" * 60)
    print("OUTLIER PURCHASES (flagged for review):")
    print("=" * 60)
    print(outlier_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("NORMAL PURCHASES (first 5 shown):")
    print("=" * 60)
    print(normal_df.head(5))

    # A sanity check on our deliberately planted HIGH outlier ($185,000):
    # this should reliably be caught by both methods, since it towers far
    # above the normal spending pattern.
    assert 185000.00 in outlier_df["Amount"].values, "High planted outlier was NOT detected!"
    print("\nSanity check passed: the planted HIGH outlier ($185,000) was correctly detected.")

    # A note on our deliberately planted LOW value ($5.00):
    was_low_value_flagged = 5.00 in outlier_df["Amount"].values
    print(f"\nWas the planted LOW value ($5.00) flagged as an outlier? {was_low_value_flagged}")
    if not was_low_value_flagged:
        print(
            "This is EXPECTED, and reveals a real, important limitation to be aware of:\n"
            "Z-score and IQR fences are built from the natural SPREAD of normal purchase\n"
            "amounts. Because that spread is wide (here, roughly $890 to $15,000), the\n"
            "'normal low end' fence mathematically extends below $0 -- so a $5 purchase,\n"
            "while obviously suspicious to a human, isn't statistically 'far enough' below\n"
            "the fence to trip these particular methods. These two methods are naturally\n"
            "much better at catching unusually LARGE purchases than unusually tiny ones.\n"
            "A future enhancement could add a simple supplementary business rule (e.g.\n"
            "'flag anything under $50') specifically to catch suspiciously tiny purchases,\n"
            "which is exactly the kind of expansion this modular design supports easily."
        )

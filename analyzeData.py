"""
=============================================================================
 analyzeData.py
=============================================================================
STEP 3 OF THE PIPELINE: SPEND ANALYSIS

This module takes the CLEANED DataFrame produced by cleanData.py and turns
it into a set of SUMMARY TABLES (small, aggregated DataFrames) that answer
the actual business questions a Clinical Engineering / procurement leader
cares about: "Who are we spending the most with?", "Is spend trending up?",
"What are our biggest single purchases?", etc.

WHY THIS IS A SEPARATE FILE FROM graph.py:
This file's ONLY job is to produce DATA (pandas DataFrames). It does not
know or care how that data will be displayed. graph.py will later take
these exact same DataFrames and turn them into charts, and report.py will
feed them to Claude to generate the executive summary narrative. Keeping
"compute the numbers" completely separate from "display the numbers" means
we can change the visuals or the AI prompt at any time WITHOUT touching a
single line of analysis logic -- and we can unit-test the numbers on their
own, with no charting library involved at all.

WHAT THIS FILE PROVIDES (public functions):
    Grouped spending summaries (one row per category):
        1. monthly_spending()        -> spend by Year + Month
        2. supplier_spending()       -> spend by Supplier/Vendor
        3. market_spending()         -> spend by Market/Product Category
        4. cost_center_spending()    -> spend by Cost Center
        5. account_spending()        -> spend by GL/Account Code
        6. requester_spending()      -> spend by Requester (person who ordered)

    Single-number / ranking metrics (each still returned as a DataFrame,
    per the requirement "return DataFrames instead of printing them"):
        7. top_purchase_orders()     -> the N largest individual purchases
        8. average_purchase_amount() -> the average PO amount, dataset-wide
        9. largest_purchase()        -> the single largest individual purchase
        10. number_of_suppliers()    -> count of distinct suppliers
        11. number_of_purchase_orders() -> count of total purchase orders

    Orchestrator:
        12. generate_all_summaries() -> runs everything at once, returns a
                                         dictionary of {name: DataFrame},
                                         skipping any summary whose required
                                         column isn't present in this dataset
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# pandas provides the DataFrame object and the groupby/aggregation
# operations that are the core of every function in this file.
import pandas as pd

# Type hints -- purely for readability, they don't change how the code runs.
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# INTERNAL HELPER FUNCTION (not part of the public API)
# ---------------------------------------------------------------------------
def _aggregate_spending_by_column(
    df: pd.DataFrame,
    group_column: str,
    amount_column: str,
    sort_descending: bool = True,
) -> pd.DataFrame:
    """
    A reusable, internal helper that groups the dataset by one column and
    computes standard spend metrics for each group.

    WHY THIS HELPER EXISTS:
    Five of the required summaries (supplier, market, cost center, account,
    requester spending) all do the EXACT same underlying computation --
    they just group by a different column. Writing this logic five
    separate times would mean five separate places to introduce a bug, and
    five separate places to update if we ever change what a "spending
    summary" includes. This helper is written once, tested once, and
    reused by every summary function below. The leading underscore in the
    function name is a Python convention signaling "this is an internal
    building block, not meant to be called directly by outside code."

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    group_column : str
        The column to group by, e.g. "Supplier", "Cost Center".
    amount_column : str
        The column containing the numeric purchase amount to aggregate.
    sort_descending : bool, default True
        Whether to sort results from highest to lowest total spend. This
        defaults to True because in almost every use case here, the
        biggest spenders/categories are the most important ones to see
        first (e.g. for an executive glancing at a table).

    Returns
    -------
    pd.DataFrame
        A summary table with one row per group and these columns:
            - the group column itself (e.g. "Supplier")
            - "Total Spend"      : sum of all purchase amounts in that group
            - "Number of Orders" : how many purchase orders fall in that group
            - "Average Order"    : Total Spend / Number of Orders for that group
            - "Percent of Total" : this group's share of TOTAL spend across
                                    the whole dataset, as a percentage
    """

    # Defensive check: fail clearly if either required column is missing,
    # rather than letting pandas raise a confusing internal KeyError.
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in dataset.")
    if amount_column not in df.columns:
        raise ValueError(f"Column '{amount_column}' not found in dataset.")

    # df.groupby(group_column) splits the DataFrame into a separate
    # mini-table for every unique value in group_column (e.g. one mini-
    # table per supplier). [amount_column] then selects just the spend
    # column from each of those mini-tables, since that's all we need to
    # aggregate. .agg([...]) then computes multiple statistics at once
    # for each group in a single pass over the data (much more efficient
    # than calling .sum(), .count(), .mean() separately).
    summary = (
        df.groupby(group_column)[amount_column]
        .agg(["sum", "count", "mean"])
        .reset_index()  # turns the group column back into a normal column
                         # instead of a special pandas "index", which makes
                         # the result easier to work with everywhere else
                         # (charts, Streamlit tables, etc.)
    )

    # Rename the generic aggregation output names ("sum", "count", "mean")
    # into clear, human-readable column names for anyone viewing this table.
    summary = summary.rename(
        columns={
            "sum": "Total Spend",
            "count": "Number of Orders",
            "mean": "Average Order",
        }
    )

    # Calculate what percentage of TOTAL spend (across every group
    # combined) each individual group represents. This turns a plain
    # dollar figure into useful context -- e.g. "$1.2M" means very
    # different things depending on whether that's 5% or 60% of all
    # procurement spend.
    total_spend_all_groups = summary["Total Spend"].sum()
    summary["Percent of Total"] = (
        (summary["Total Spend"] / total_spend_all_groups) * 100
    ).round(2)

    # Sort so the biggest-spend groups appear first, which is almost
    # always what a reader wants to see at the top of a summary table.
    if sort_descending:
        summary = summary.sort_values(by="Total Spend", ascending=False)

    # Reset the row index (0, 1, 2, ...) after sorting, since sorting
    # leaves the original (now out-of-order) index numbers behind
    # otherwise, which looks confusing in a displayed table.
    summary = summary.reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------------
# FUNCTION 1: monthly_spending
# ---------------------------------------------------------------------------
def monthly_spending(
    df: pd.DataFrame,
    amount_column: str,
    year_column: str = "Year",
    month_column: str = "Month",
) -> pd.DataFrame:
    """
    Summarize total procurement spend for each Year-Month combination.

    REASONING -- why this metric matters:
    Monthly spend is the single most important view for spotting TRENDS:
    is procurement spend rising, falling, or seasonal? A Clinical
    Engineering leader needs to know if spend is spiking (e.g. due to
    equipment failures requiring emergency replacement) or following a
    predictable annual pattern (e.g. higher spend near fiscal year-end due
    to "use it or lose it" budgets). This table is also the direct input
    for the monthly trend line chart built later in graph.py.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset (must already have Year and Month columns,
        created by cleanData.py's create_date_features()).
    amount_column : str
        Name of the numeric purchase amount column.
    year_column : str, default "Year"
        Name of the year column.
    month_column : str, default "Month"
        Name of the month column.

    Returns
    -------
    pd.DataFrame
        One row per Year-Month, sorted chronologically, with columns:
        Year, Month, Total Spend, Number of Orders, Average Order.
    """

    for col in (amount_column, year_column, month_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    # Group by BOTH Year and Month together (passed as a list). This
    # matters because grouping by Month alone would incorrectly combine
    # "January 2024" and "January 2025" into a single "January" bucket --
    # grouping by both ensures each calendar month in each specific year
    # gets its own row.
    summary = (
        df.groupby([year_column, month_column])[amount_column]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(
            columns={
                "sum": "Total Spend",
                "count": "Number of Orders",
                "mean": "Average Order",
            }
        )
    )

    # Sort chronologically (by Year, then Month within each year) rather
    # than by spend amount -- for a TIME-BASED table, reading it in date
    # order is what makes a trend visible at a glance.
    summary = summary.sort_values(by=[year_column, month_column]).reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------------
# FUNCTION 2: supplier_spending
# ---------------------------------------------------------------------------
def supplier_spending(
    df: pd.DataFrame,
    amount_column: str,
    supplier_column: str,
) -> pd.DataFrame:
    """
    Summarize total procurement spend per supplier/vendor.

    REASONING -- why this metric matters:
    Vendor concentration is one of the most important risk and negotiating
    signals in procurement. If a small number of suppliers account for the
    majority of spend, that creates:
        - Negotiating leverage opportunities (large-volume vendors are
          often willing to offer better pricing/contract terms)
        - Supply-chain RISK (if a top supplier has an outage, price hike,
          or goes out of business, a large share of purchasing is
          disrupted)
    This table directly powers the "Top Vendors by Spend" chart and is
    typically the first table an executive summary highlights.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    supplier_column : str
        Name of the supplier/vendor name column (should already be
        standardized by cleanData.py's standardize_supplier_names()).

    Returns
    -------
    pd.DataFrame
        One row per supplier, sorted by Total Spend descending.
    """

    return _aggregate_spending_by_column(df, supplier_column, amount_column)


# ---------------------------------------------------------------------------
# FUNCTION 3: market_spending
# ---------------------------------------------------------------------------
def market_spending(
    df: pd.DataFrame,
    amount_column: str,
    market_column: str,
) -> pd.DataFrame:
    """
    Summarize total procurement spend per Market / product category
    (e.g. "Imaging Equipment", "Surgical Instruments", "IT Hardware").

    REASONING -- why this metric matters:
    While supplier spending shows WHO money goes to, market/category
    spending shows WHAT it's being spent ON. This is essential for
    category-level cost-reduction strategy (e.g. "our imaging equipment
    category has grown 30% year over year -- why?") and for identifying
    which clinical equipment categories are the biggest budget drivers,
    independent of which specific vendor supplies them.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    market_column : str
        Name of the market/category column.

    Returns
    -------
    pd.DataFrame
        One row per market/category, sorted by Total Spend descending.
    """

    return _aggregate_spending_by_column(df, market_column, amount_column)


# ---------------------------------------------------------------------------
# FUNCTION 4: cost_center_spending
# ---------------------------------------------------------------------------
def cost_center_spending(
    df: pd.DataFrame,
    amount_column: str,
    cost_center_column: str,
) -> pd.DataFrame:
    """
    Summarize total procurement spend per Cost Center (i.e. per hospital
    department/unit responsible for the budget).

    REASONING -- why this metric matters:
    Cost centers are how hospitals track BUDGET ACCOUNTABILITY -- each
    department is typically allocated (and monitored against) its own
    budget. This table lets Clinical Engineering leadership see which
    departments/units are driving the most spend, supports budget
    compliance conversations ("Cardiology is running 20% over their
    typical monthly cost center spend"), and helps allocate future budget
    planning based on real historical usage per unit.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    cost_center_column : str
        Name of the cost center column.

    Returns
    -------
    pd.DataFrame
        One row per cost center, sorted by Total Spend descending.
    """

    return _aggregate_spending_by_column(df, cost_center_column, amount_column)


# ---------------------------------------------------------------------------
# FUNCTION 5: account_spending
# ---------------------------------------------------------------------------
def account_spending(
    df: pd.DataFrame,
    amount_column: str,
    account_column: str,
) -> pd.DataFrame:
    """
    Summarize total procurement spend per GL / accounting code.

    REASONING -- why this metric matters:
    The "Account" (often a General Ledger / GL code) is how the FINANCE
    department classifies spend for accounting and reporting purposes,
    separate from how Clinical Engineering thinks about it operationally.
    This table is essential for financial reconciliation ("does our
    procurement data match what Finance sees in the GL?"), audit trails,
    and month-end/year-end close processes where spend must be
    categorized correctly by account code.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    account_column : str
        Name of the account/GL code column.

    Returns
    -------
    pd.DataFrame
        One row per account code, sorted by Total Spend descending.
    """

    return _aggregate_spending_by_column(df, account_column, amount_column)


# ---------------------------------------------------------------------------
# FUNCTION 6: requester_spending
# ---------------------------------------------------------------------------
def requester_spending(
    df: pd.DataFrame,
    amount_column: str,
    requester_column: str,
) -> pd.DataFrame:
    """
    Summarize total procurement spend per Requester (the individual person
    or role who submitted the purchase request).

    REASONING -- why this metric matters:
    Requester-level spend is important for INTERNAL CONTROL and workflow
    review: it flags which individuals or roles are generating the most
    purchase volume, which supports:
        - Approval workflow review (does high-volume requesting align with
          who actually has purchasing authority?)
        - Spotting potential "purchase order splitting" -- a known
          compliance red flag where a large purchase is broken into
          several smaller POs to stay under an approval-threshold
          (this becomes visible when one requester has an unusually high
          NUMBER of orders relative to their average order size)
        - Training/support needs (e.g. a requester's average order amount
          being an outlier relative to peers may indicate confusion about
          proper purchasing categories)

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    requester_column : str
        Name of the requester column.

    Returns
    -------
    pd.DataFrame
        One row per requester, sorted by Total Spend descending.
    """

    return _aggregate_spending_by_column(df, requester_column, amount_column)


# ---------------------------------------------------------------------------
# FUNCTION 7: top_purchase_orders
# ---------------------------------------------------------------------------
def top_purchase_orders(
    df: pd.DataFrame,
    amount_column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the N single largest individual purchase orders in the dataset.

    REASONING -- why this metric matters:
    Aggregated summaries (like supplier or cost center spending) can hide
    important individual transactions -- a department's total spend might
    look normal, but could be driven by one enormous, unusual purchase.
    Reviewing the largest individual purchase orders directly is typically
    the FIRST thing a finance director or auditor wants to see, since
    these are the transactions most worth double-checking for accuracy,
    approval compliance, or negotiation opportunities (large one-off buys
    are often where the best discount negotiations are possible).

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    top_n : int, default 10
        How many of the largest purchase orders to return.

    Returns
    -------
    pd.DataFrame
        The top_n rows of the ORIGINAL dataset (all original columns kept,
        e.g. PO Number, Supplier, Date), sorted by amount descending, so
        the reader has full context on each large purchase, not just its
        dollar value.
    """

    if amount_column not in df.columns:
        raise ValueError(f"Column '{amount_column}' not found in dataset.")

    # sort_values(ascending=False) orders every row from largest to
    # smallest purchase amount. .head(top_n) then keeps only the first
    # N rows of that sorted result -- i.e. the N biggest purchases.
    top_orders = (
        df.sort_values(by=amount_column, ascending=False)
        .head(top_n)
        .reset_index(drop=True)  # tidy row numbering (0, 1, 2, ...) for display
    )

    return top_orders


# ---------------------------------------------------------------------------
# FUNCTION 8: average_purchase_amount
# ---------------------------------------------------------------------------
def average_purchase_amount(df: pd.DataFrame, amount_column: str) -> pd.DataFrame:
    """
    Calculate the average (mean) purchase order amount across the ENTIRE
    dataset.

    REASONING -- why this metric matters:
    The average purchase amount is the single reference number that
    defines what a "typical" purchase looks like for this organization.
    It's a baseline used in two important ways:
        1. Context for every other number in this analysis -- e.g. knowing
           the average PO is $2,300 makes a $45,000 purchase obviously
           notable at a glance.
        2. A foundational input for outlier detection (outlierDetect.py),
           where purchases are flagged as unusual specifically because
           they deviate significantly from this average (and the spread
           around it).

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.

    Returns
    -------
    pd.DataFrame
        A ONE-ROW DataFrame (per this project's requirement to always
        return DataFrames, never bare numbers or printed text) with a
        single column "Average Purchase Amount".
    """

    if amount_column not in df.columns:
        raise ValueError(f"Column '{amount_column}' not found in dataset.")

    # .mean() computes the average of every value in the amount column.
    average_value = df[amount_column].mean()

    # pd.DataFrame([{...}]) builds a DataFrame from a list containing one
    # dictionary -- this is a simple, readable way to create a DataFrame
    # that has exactly one row.
    return pd.DataFrame([{"Average Purchase Amount": round(average_value, 2)}])


# ---------------------------------------------------------------------------
# FUNCTION 9: largest_purchase
# ---------------------------------------------------------------------------
def largest_purchase(df: pd.DataFrame, amount_column: str) -> pd.DataFrame:
    """
    Return the single largest individual purchase order in the dataset,
    with all of its original details.

    REASONING -- why this metric matters:
    This is the fastest possible "sanity check" and headline data point in
    any spend analysis: what is the single biggest thing we bought, from
    whom, and when? It's often the very first fact cited in an executive
    summary, and it's also a quick way to catch a data-entry error (e.g. a
    purchase accidentally entered as $4,500,000 instead of $4,500 will
    immediately stand out here).

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame containing the full record (all original
        columns) of the single largest purchase order.
    """

    if amount_column not in df.columns:
        raise ValueError(f"Column '{amount_column}' not found in dataset.")

    # .idxmax() finds the ROW LABEL (index) of the maximum value in the
    # amount column -- i.e. "which row has the biggest purchase amount?"
    largest_row_index = df[amount_column].idxmax()

    # .loc[[largest_row_index]] retrieves that specific row. We wrap the
    # index in double brackets ([[ ]]) rather than single brackets so the
    # result is a DataFrame (one row) rather than a Series (which would
    # lose the tabular column structure) -- keeping the "always return a
    # DataFrame" rule consistent everywhere in this module.
    return df.loc[[largest_row_index]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FUNCTION 10: number_of_suppliers
# ---------------------------------------------------------------------------
def number_of_suppliers(df: pd.DataFrame, supplier_column: str) -> pd.DataFrame:
    """
    Count how many distinct/unique suppliers appear in the dataset.

    REASONING -- why this metric matters:
    The size of a vendor base is a strategic indicator on its own. A
    procurement operation with very FEW suppliers may have strong
    volume-based pricing but is exposed to vendor lock-in and
    single-point-of-failure supply risk. A procurement operation with an
    unusually LARGE number of suppliers may be missing out on volume
    discounts and could benefit from vendor consolidation. This single
    number is a common headline figure in a procurement executive summary.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    supplier_column : str
        Name of the supplier/vendor name column.

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame with a single column "Number of Suppliers".
    """

    if supplier_column not in df.columns:
        raise ValueError(f"Column '{supplier_column}' not found in dataset.")

    # .nunique() counts how many DISTINCT values exist in the column,
    # automatically ignoring missing values (NaN) so they don't get
    # miscounted as a fake "supplier."
    supplier_count = df[supplier_column].nunique()

    return pd.DataFrame([{"Number of Suppliers": supplier_count}])


# ---------------------------------------------------------------------------
# FUNCTION 11: number_of_purchase_orders
# ---------------------------------------------------------------------------
def number_of_purchase_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the total number of purchase order rows/transactions in the
    dataset.

    REASONING -- why this metric matters:
    Order VOLUME (as opposed to dollar amount) is an important
    operational workload indicator: it tells the procurement/Clinical
    Engineering office how many individual transactions they are
    processing, which relates directly to staffing needs, processing time,
    and administrative overhead -- independent of how much money is
    involved. A high order count with a low average order amount (see
    average_purchase_amount) may suggest an opportunity to consolidate
    many small orders into fewer, larger ones for efficiency.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame with a single column "Number of Purchase Orders".
    """

    # len(df) simply counts how many rows are in the DataFrame -- since
    # each row represents one purchase order after cleaning, this is a
    # direct count of total purchase order transactions.
    return pd.DataFrame([{"Number of Purchase Orders": len(df)}])


# ---------------------------------------------------------------------------
# FUNCTION 12: generate_all_summaries (the orchestrator)
# ---------------------------------------------------------------------------
def generate_all_summaries(
    df: pd.DataFrame,
    amount_column: str,
    supplier_column: Optional[str] = None,
    market_column: Optional[str] = None,
    cost_center_column: Optional[str] = None,
    account_column: Optional[str] = None,
    requester_column: Optional[str] = None,
    year_column: str = "Year",
    month_column: str = "Month",
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Run every analysis function in this module and collect the results
    into one dictionary.

    WHY THIS FUNCTION EXISTS:
    Not every hospital department's Excel export will have every optional
    column (e.g. some datasets may not track "Requester" or "Market").
    Rather than force every summary to run and crash on a missing column,
    this orchestrator only runs a summary if its required column was
    actually supplied AND exists in the dataset -- and it records which
    summaries were skipped and why, so the calling code (main.py) can
    display a clear message instead of a confusing crash.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    amount_column : str
        Name of the numeric purchase amount column (required for almost
        every summary).
    supplier_column, market_column, cost_center_column, account_column,
    requester_column : str, optional
        Names of the respective optional columns. Pass None (the default)
        for any column that doesn't exist in this particular dataset --
        that summary will simply be skipped.
    year_column, month_column : str
        Names of the Year/Month columns created by cleanData.py's
        create_date_features(), used for monthly_spending().
    top_n : int, default 10
        How many rows to include in the top_purchase_orders() table.

    Returns
    -------
    Dict[str, Any]
        A dictionary shaped like:
            {
                "monthly_spending": <DataFrame>,
                "supplier_spending": <DataFrame> (or skipped),
                ...
                "skipped_summaries": {"market_spending": "reason...", ...}
            }
    """

    results: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}

    # --- Monthly spending (always attempted -- Year/Month should always
    # exist after cleanData.py's create_date_features() has run) ---
    try:
        results["monthly_spending"] = monthly_spending(
            df, amount_column, year_column, month_column
        )
    except ValueError as error:
        skipped["monthly_spending"] = str(error)

    # --- Optional groupby summaries: each is only run if its column name
    # was provided by the caller AND that column actually exists. This
    # pattern is repeated for each optional dimension. ---
    optional_summaries = {
        "supplier_spending": (supplier_column, supplier_spending),
        "market_spending": (market_column, market_spending),
        "cost_center_spending": (cost_center_column, cost_center_spending),
        "account_spending": (account_column, account_spending),
        "requester_spending": (requester_column, requester_spending),
    }

    for summary_name, (column_name, summary_function) in optional_summaries.items():
        if column_name is None:
            skipped[summary_name] = "No column name was provided for this summary."
        elif column_name not in df.columns:
            skipped[summary_name] = f"Column '{column_name}' not found in dataset."
        else:
            results[summary_name] = summary_function(df, amount_column, column_name)

    # --- Metrics that don't depend on any optional column, so these
    # always run. ---
    results["top_purchase_orders"] = top_purchase_orders(df, amount_column, top_n)
    results["average_purchase_amount"] = average_purchase_amount(df, amount_column)
    results["largest_purchase"] = largest_purchase(df, amount_column)
    results["number_of_purchase_orders"] = number_of_purchase_orders(df)

    # number_of_suppliers still needs a supplier column, so it follows the
    # same "skip gracefully" pattern as the optional summaries above.
    if supplier_column is not None and supplier_column in df.columns:
        results["number_of_suppliers"] = number_of_suppliers(df, supplier_column)
    else:
        skipped["number_of_suppliers"] = "No valid supplier column was provided."

    # Attach the skip log so the caller can see exactly what was left out
    # and why, instead of silently missing data with no explanation.
    results["skipped_summaries"] = skipped

    return results


# ---------------------------------------------------------------------------
# STANDALONE TEST BLOCK
# ---------------------------------------------------------------------------
# Runs only when you execute this file directly: python src/analyzeData.py
# We build a small, already-clean sample dataset (as if it just came out of
# cleanData.py) and run every summary function against it to prove each
# one works and returns a proper DataFrame.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    sample_clean_data = {
        "PO Number": [f"PO-{1000+i}" for i in range(12)],
        "Supplier": [
            "Medtronic", "Ge Healthcare", "Stryker", "Medtronic",
            "Philips", "Stryker", "Medtronic", "Ge Healthcare",
            "Philips", "Stryker", "Medtronic", "Ge Healthcare",
        ],
        "Market": [
            "Patient Monitoring", "Imaging", "Surgical", "Patient Monitoring",
            "Imaging", "Surgical", "Patient Monitoring", "Imaging",
            "Imaging", "Surgical", "Patient Monitoring", "Imaging",
        ],
        "Cost Center": [
            "ICU", "Radiology", "OR", "ICU", "Radiology", "OR",
            "ICU", "Radiology", "Radiology", "OR", "ICU", "Radiology",
        ],
        "Account": [
            "6100", "6200", "6300", "6100", "6200", "6300",
            "6100", "6200", "6200", "6300", "6100", "6200",
        ],
        "Requester": [
            "J. Smith", "A. Lee", "M. Chen", "J. Smith", "A. Lee",
            "M. Chen", "J. Smith", "A. Lee", "A. Lee", "M. Chen",
            "J. Smith", "A. Lee",
        ],
        "PO Amount Ordered": [
            4500.00, 12800.50, 3200.00, 4600.00, 980.25, 3100.00,
            4550.00, 15000.00, 890.00, 3050.00, 47000.00, 13200.00,
        ],
        "PO Date": pd.to_datetime([
            "2024-01-05", "2024-01-12", "2024-01-20", "2024-02-10",
            "2024-02-15", "2024-02-22", "2024-03-01", "2024-03-10",
            "2024-03-18", "2024-04-02", "2024-04-09", "2024-04-20",
        ]),
    }
    clean_df = pd.DataFrame(sample_clean_data)
    clean_df["Year"] = clean_df["PO Date"].dt.year
    clean_df["Month"] = clean_df["PO Date"].dt.month

    print("=" * 60)
    print("MONTHLY SPENDING")
    print("=" * 60)
    print(monthly_spending(clean_df, "PO Amount Ordered"))

    print("\n" + "=" * 60)
    print("SUPPLIER SPENDING")
    print("=" * 60)
    print(supplier_spending(clean_df, "PO Amount Ordered", "Supplier"))

    print("\n" + "=" * 60)
    print("MARKET SPENDING")
    print("=" * 60)
    print(market_spending(clean_df, "PO Amount Ordered", "Market"))

    print("\n" + "=" * 60)
    print("COST CENTER SPENDING")
    print("=" * 60)
    print(cost_center_spending(clean_df, "PO Amount Ordered", "Cost Center"))

    print("\n" + "=" * 60)
    print("ACCOUNT SPENDING")
    print("=" * 60)
    print(account_spending(clean_df, "PO Amount Ordered", "Account"))

    print("\n" + "=" * 60)
    print("REQUESTER SPENDING")
    print("=" * 60)
    print(requester_spending(clean_df, "PO Amount Ordered", "Requester"))

    print("\n" + "=" * 60)
    print("TOP 5 PURCHASE ORDERS")
    print("=" * 60)
    print(top_purchase_orders(clean_df, "PO Amount Ordered", top_n=5))

    print("\n" + "=" * 60)
    print("AVERAGE PURCHASE AMOUNT")
    print("=" * 60)
    print(average_purchase_amount(clean_df, "PO Amount Ordered"))

    print("\n" + "=" * 60)
    print("LARGEST PURCHASE")
    print("=" * 60)
    print(largest_purchase(clean_df, "PO Amount Ordered"))

    print("\n" + "=" * 60)
    print("NUMBER OF SUPPLIERS")
    print("=" * 60)
    print(number_of_suppliers(clean_df, "Supplier"))

    print("\n" + "=" * 60)
    print("NUMBER OF PURCHASE ORDERS")
    print("=" * 60)
    print(number_of_purchase_orders(clean_df))

    print("\n" + "=" * 60)
    print("generate_all_summaries() -- full orchestrator test")
    print("=" * 60)
    all_summaries = generate_all_summaries(
        clean_df,
        amount_column="PO Amount Ordered",
        supplier_column="Supplier",
        market_column="Market",
        cost_center_column="Cost Center",
        account_column="Account",
        requester_column="Requester",
    )
    print(f"Summaries generated: {[k for k in all_summaries.keys() if k != 'skipped_summaries']}")
    print(f"Skipped summaries: {all_summaries['skipped_summaries']}")

"""
=============================================================================
 graph.py
=============================================================================
STEP 4 OF THE PIPELINE: INTERACTIVE VISUALIZATION

This module takes the summary DataFrames produced by analyzeData.py (and,
for a couple of charts, the cleaned raw DataFrame from cleanData.py) and
turns them into interactive Plotly charts, styled to look like a
professional procurement analytics report (in the spirit of Oracle
Procurement Cloud dashboards) rather than a default/generic chart.

WHY THIS FILE ONLY RETURNS FIGURES (never displays them):
Every function in this file returns a plotly.graph_objects.Figure object
and NEVER calls .show(). This is intentional: main.py's Streamlit app is
the only place that should decide HOW and WHERE a figure gets displayed
(e.g. st.plotly_chart(fig)). It's also what allows graph.py to be tested
completely on its own (see the bottom of this file) without opening a
browser window, and lets the same figures later be saved as image/HTML
files into the graphs/ folder if desired.

WHAT THIS FILE PROVIDES (public functions):
    1. monthly_spending_line_chart()
    2. supplier_spending_bar_chart()
    3. market_spending_pie_chart()
    4. cost_center_spending_bar_chart()
    5. account_spending_pie_chart()
    6. top_suppliers_bar_chart()
    7. monthly_spending_by_supplier_stacked_bar_chart()
    8. purchase_amount_distribution_histogram()
    9. outlier_scatter_plot()
    10. spending_trend_dashboard()
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# pandas is needed here because a couple of these chart functions must
# group/pivot the RAW cleaned dataset themselves (e.g. the stacked bar
# chart needs spend broken down by BOTH month AND supplier at once, which
# isn't one of the pre-built summary tables from analyzeData.py).
import pandas as pd

# plotly.graph_objects (go) gives us low-level, precise control over
# individual chart traces (bars, lines, pie slices) -- we use this for
# most charts so we can fine-tune hover text, colors, and layout exactly.
import plotly.graph_objects as go

# make_subplots lets us combine multiple individual charts into one
# multi-panel "dashboard" figure (used in spending_trend_dashboard()).
from plotly.subplots import make_subplots

# Type hints -- purely for readability, they don't change how the code runs.
from typing import Optional, List


# ---------------------------------------------------------------------------
# SHARED VISUAL THEME ("Clean Healthcare Analytics" look)
# ---------------------------------------------------------------------------
# Defining the color palette and fonts ONCE, here, at the top of the file,
# means every chart in this module automatically looks like part of the
# same professional report -- instead of 10 charts each with Plotly's
# random default colors. This is the same idea as a hospital's brand
# style guide: consistent colors and fonts build trust and readability.
# ---------------------------------------------------------------------------

# A calm, professional palette: deep navy and teal (common in healthcare/
# clinical branding) plus supporting slate and muted accent tones. This
# ordered list is used whenever a chart has multiple categories (bars,
# pie slices, stacked segments) -- Plotly assigns colors in this order.
HEALTHCARE_COLOR_PALETTE: List[str] = [
    "#0B3C5D",  # deep navy      - primary
    "#1F8A70",  # teal green     - secondary
    "#328CC1",  # medium blue    - tertiary
    "#D9B44A",  # muted gold     - accent
    "#6C7A89",  # slate gray     - neutral
    "#A23B72",  # muted plum     - additional category
    "#5A9367",  # soft sage      - additional category
    "#8E5572",  # dusty mauve    - additional category
]

# A dedicated color for flagging OUTLIERS/anomalies. Red is a near-universal
# "pay attention here" signal, reserved ONLY for this purpose so it always
# stands out against the calmer palette above.
OUTLIER_COLOR = "#C0392B"
NORMAL_POINT_COLOR = "#328CC1"

# The font family used across every chart -- a clean, modern sans-serif
# that renders consistently across operating systems and browsers.
FONT_FAMILY = "Segoe UI, Arial, sans-serif"


def _apply_healthcare_theme(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    """
    Apply the shared "clean healthcare analytics" visual theme to any
    Plotly figure: consistent fonts, colors, background, and title
    styling.

    WHY THIS HELPER EXISTS:
    Every chart-building function below creates its own figure, but they
    should all SHARE the same professional look and feel. Rather than
    repeating the same ~15 lines of layout styling in all 10 functions
    (and risking them drifting out of sync if we tweak the theme later),
    we centralize it here and call this one helper at the end of every
    chart function.

    Parameters
    ----------
    fig : go.Figure
        The figure to style (already containing its data/traces).
    title : str
        The chart's main title.
    x_title : str, optional
        Label for the x-axis (leave blank for charts without a meaningful
        axis, e.g. pie charts).
    y_title : str, optional
        Label for the y-axis.

    Returns
    -------
    go.Figure
        The same figure object, with the shared theme applied (styling is
        applied in place, and the figure is also returned for convenience
        so calls can be chained).
    """

    fig.update_layout(
        # A clear, bold title at the top, left-aligned like a real report
        # heading rather than centered like a casual chart.
        title={
            "text": title,
            "font": {"size": 20, "family": FONT_FAMILY, "color": "#0B3C5D"},
            "x": 0.02,
            "xanchor": "left",
        },
        # Plain white backgrounds (both the chart area and the whole
        # figure) look clean and print well in reports -- avoiding
        # Plotly's default gray gridlines and shading.
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": FONT_FAMILY, "size": 13, "color": "#333333"},
        # A legend placed neatly along the top, only shown when a chart
        # actually has multiple categories to distinguish.
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        # Comfortable margins so titles/labels are never clipped.
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        # Assign our custom color palette as the default sequence for any
        # trace that doesn't explicitly set its own colors.
        colorway=HEALTHCARE_COLOR_PALETTE,
        # A styled hover tooltip (the box that appears when you mouse over
        # a data point) -- dark background with white text reads clearly.
        hoverlabel={"bgcolor": "#0B3C5D", "font": {"color": "white", "family": FONT_FAMILY}},
    )

    # Light, subtle gridlines only on the y-axis (common in professional
    # financial reports) -- this guides the eye without cluttering the
    # chart the way full gridlines on both axes would.
    fig.update_yaxes(title_text=y_title, gridcolor="#E5E5E5", zeroline=False)
    fig.update_xaxes(title_text=x_title, showgrid=False)

    return fig


# ---------------------------------------------------------------------------
# CHART 1: monthly_spending_line_chart
# ---------------------------------------------------------------------------
def monthly_spending_line_chart(
    monthly_summary_df: pd.DataFrame,
    year_column: str = "Year",
    month_column: str = "Month",
    spend_column: str = "Total Spend",
) -> go.Figure:
    """
    Build an interactive line chart showing total procurement spend over
    time, one point per Year-Month.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    This is typically the FIRST chart a department director looks at: is
    spend trending up, down, or following a seasonal pattern? A steadily
    climbing line may prompt a budget review; a sudden spike in one month
    may point to an equipment failure requiring emergency replacement
    purchases; a recurring spike every March/April may reveal a
    fiscal-year-end "use it or lose it" budgeting pattern worth planning
    around next year.

    Parameters
    ----------
    monthly_summary_df : pd.DataFrame
        The output of analyzeData.py's monthly_spending() function.
    year_column, month_column, spend_column : str
        Column names to use for the x-axis period and y-axis spend value.

    Returns
    -------
    go.Figure
        An interactive Plotly line chart.
    """

    df = monthly_summary_df.copy()

    # Build a clean, human-readable "Jan 2024" style label for the x-axis
    # by combining Year and Month into one text column. Without this, the
    # x-axis would either show two separate confusing columns or numeric
    # month values (1-12) with no year context.
    df["Period"] = (
        pd.to_datetime(df[year_column].astype(str) + "-" + df[month_column].astype(str) + "-01")
        .dt.strftime("%b %Y")
    )

    fig = go.Figure()

    # go.Scatter with mode="lines+markers" draws BOTH the connecting line
    # (to show the trend) AND a visible dot at each actual data point (so
    # individual months are still clearly identifiable, not just implied
    # by the line).
    fig.add_trace(
        go.Scatter(
            x=df["Period"],
            y=df[spend_column],
            mode="lines+markers",
            line={"color": HEALTHCARE_COLOR_PALETTE[0], "width": 3},
            marker={"size": 8, "color": HEALTHCARE_COLOR_PALETTE[0]},
            # hovertemplate customizes exactly what text appears when a
            # user hovers over a point. "%{y:$,.2f}" formats the number
            # as currency with commas and two decimal places.
            hovertemplate="<b>%{x}</b><br>Total Spend: %{y:$,.2f}<extra></extra>",
            name="Monthly Spend",
        )
    )

    return _apply_healthcare_theme(
        fig, title="Monthly  Spending Trend", x_title="Month", y_title="Total Spend ($)"
    )


# ---------------------------------------------------------------------------
# CHART 2: supplier_spending_bar_chart
# ---------------------------------------------------------------------------
def supplier_spending_bar_chart(
    supplier_summary_df: pd.DataFrame,
    supplier_column: str,
    spend_column: str = "Total Spend",
) -> go.Figure:
    """
    Build a horizontal bar chart showing total spend per supplier.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    A horizontal layout (rather than vertical bars) is used specifically
    because supplier names are often long text ("GE Healthcare",
    "Stryker Corporation") that would be squeezed and hard to read as
    vertical axis labels. This chart immediately shows vendor
    concentration: a small number of very long bars versus many similar-
    length bars tells the department whether they are dependent on a few
    key suppliers (negotiating leverage, but also single-point-of-failure
    risk) or spread across many vendors.

    Parameters
    ----------
    supplier_summary_df : pd.DataFrame
        The output of analyzeData.py's supplier_spending() function.
    supplier_column : str
        Name of the column containing supplier names.
    spend_column : str, default "Total Spend"
        Name of the column containing each supplier's total spend.

    Returns
    -------
    go.Figure
        An interactive Plotly horizontal bar chart, sorted so the
        largest supplier appears at the TOP (a common convention in
        ranked bar charts).
    """

    # Plotly draws horizontal bars from bottom to top by default, so to
    # make the LARGEST supplier appear at the TOP of the chart (the
    # natural reading order for a ranked list), we sort ascending here --
    # the last row drawn ends up visually on top.
    df = supplier_summary_df.sort_values(by=spend_column, ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df[spend_column],
            y=df[supplier_column],
            orientation="h",  # "h" makes this a HORIZONTAL bar chart
            marker={"color": HEALTHCARE_COLOR_PALETTE[0]},
            hovertemplate="<b>%{y}</b><br>Total Spend: %{x:$,.2f}<extra></extra>",
        )
    )

    return _apply_healthcare_theme(
        fig, title="Total Spending by Supplier", x_title="Total Spend ($)", y_title=""
    )


# ---------------------------------------------------------------------------
# CHART 3: market_spending_pie_chart
# ---------------------------------------------------------------------------
def market_spending_pie_chart(
    market_summary_df: pd.DataFrame,
    market_column: str,
    spend_column: str = "Total Spend",
) -> go.Figure:
    """
    Build a donut (ring-style pie) chart showing the share of total spend
    that belongs to each Market/product category.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    While the bar charts above show absolute dollar amounts, a pie/donut
    chart is the clearest way to show PROPORTIONS -- "what percentage of
    our total budget goes to Imaging Equipment vs. Surgical Instruments
    vs. Patient Monitoring?" This framing is often more useful for
    strategic budget conversations ("Imaging is 40% of our spend -- should
    that be our primary cost-reduction focus?") than raw dollar figures
    alone.

    Parameters
    ----------
    market_summary_df : pd.DataFrame
        The output of analyzeData.py's market_spending() function.
    market_column : str
        Name of the column containing market/category names.
    spend_column : str, default "Total Spend"
        Name of the column containing each category's total spend.

    Returns
    -------
    go.Figure
        An interactive Plotly donut chart.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=market_summary_df[market_column],
            values=market_summary_df[spend_column],
            # hole=0.45 turns a solid pie into a "donut" shape, which
            # looks more modern and leaves room in the center for a total
            # (Plotly doesn't auto-add this; it's a nice-to-have that
            # could be added later via an annotation).
            hole=0.45,
            marker={"colors": HEALTHCARE_COLOR_PALETTE},
            # textinfo controls what's written directly on each slice --
            # showing both the category name and percentage keeps the
            # chart informative without needing to hover.
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Total Spend: %{value:$,.2f}<br>Share: %{percent}<extra></extra>",
        )
    )

    return _apply_healthcare_theme(fig, title="Spending Share by Market/Category")


# ---------------------------------------------------------------------------
# CHART 4: cost_center_spending_bar_chart
# ---------------------------------------------------------------------------
def cost_center_spending_bar_chart(
    cost_center_summary_df: pd.DataFrame,
    cost_center_column: str,
    spend_column: str = "Total Spend",
) -> go.Figure:
    """
    Build a vertical bar chart showing total spend per hospital cost
    center (department/unit).

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    Cost centers are usually short, comparable labels (e.g. "ICU", "OR",
    "Radiology"), so a standard VERTICAL bar chart reads naturally here
    (unlike suppliers, which often have long names better suited to
    horizontal bars). This chart directly supports budget accountability
    conversations: which departments are consuming the most procurement
    budget, and does that align with expected clinical activity/volume in
    those units?

    Parameters
    ----------
    cost_center_summary_df : pd.DataFrame
        The output of analyzeData.py's cost_center_spending() function.
    cost_center_column : str
        Name of the column containing cost center names.
    spend_column : str, default "Total Spend"
        Name of the column containing each cost center's total spend.

    Returns
    -------
    go.Figure
        An interactive Plotly vertical bar chart, sorted largest to
        smallest, left to right.
    """

    # Sort descending so the biggest-spending cost center appears first
    # (leftmost), matching how a reader naturally scans left-to-right.
    df = cost_center_summary_df.sort_values(by=spend_column, ascending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df[cost_center_column],
            y=df[spend_column],
            marker={"color": HEALTHCARE_COLOR_PALETTE[1]},
            hovertemplate="<b>%{x}</b><br>Total Spend: %{y:$,.2f}<extra></extra>",
        )
    )

    return _apply_healthcare_theme(
        fig, title="Total Spending by Cost Center", x_title="Cost Center", y_title="Total Spend ($)"
    )


# ---------------------------------------------------------------------------
# CHART 5: account_spending_pie_chart
# ---------------------------------------------------------------------------
def account_spending_pie_chart(
    account_summary_df: pd.DataFrame,
    account_column: str,
    spend_column: str = "Total Spend",
) -> go.Figure:
    """
    Build a donut chart showing the share of total spend attributed to
    each GL/accounting code.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    This chart bridges Clinical Engineering's operational view of
    procurement with how the Finance department categorizes the exact
    same spend for accounting purposes. It's especially useful during
    month-end/year-end close or audit prep, when someone needs a quick
    visual sanity check that spend is distributed across account codes
    the way Finance expects, and to spot if one account code has grown to
    an unexpectedly large share of the total.

    Parameters
    ----------
    account_summary_df : pd.DataFrame
        The output of analyzeData.py's account_spending() function.
    account_column : str
        Name of the column containing account/GL codes.
    spend_column : str, default "Total Spend"
        Name of the column containing each account's total spend.

    Returns
    -------
    go.Figure
        An interactive Plotly donut chart.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=account_summary_df[account_column],
            values=account_summary_df[spend_column],
            hole=0.45,
            marker={"colors": HEALTHCARE_COLOR_PALETTE},
            textinfo="label+percent",
            hovertemplate="<b>Account %{label}</b><br>Total Spend: %{value:$,.2f}<br>Share: %{percent}<extra></extra>",
        )
    )

    return _apply_healthcare_theme(fig, title="Spending Share by Account / GL Code")


# ---------------------------------------------------------------------------
# CHART 6: top_suppliers_bar_chart
# ---------------------------------------------------------------------------
def top_suppliers_bar_chart(
    supplier_summary_df: pd.DataFrame,
    supplier_column: str,
    spend_column: str = "Total Spend",
    top_n: int = 15,
) -> go.Figure:
    """
    Build a horizontal bar chart showing ONLY the top N suppliers by
    total spend (default 15).

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    supplier_spending_bar_chart() (chart #2) shows EVERY supplier, which
    can become cluttered and hard to read if a hospital works with 50+
    vendors. This chart deliberately narrows the focus to the suppliers
    that matter most for strategic conversations -- vendor negotiation
    prep, consolidation opportunities, and concentration-risk review are
    almost always focused on the top vendors, not the long tail of
    suppliers used once or twice a year.

    Parameters
    ----------
    supplier_summary_df : pd.DataFrame
        The output of analyzeData.py's supplier_spending() function
        (should already be sorted descending by spend, but this function
        re-sorts defensively regardless).
    supplier_column : str
        Name of the column containing supplier names.
    spend_column : str, default "Total Spend"
        Name of the column containing each supplier's total spend.
    top_n : int, default 15
        How many top suppliers to include.

    Returns
    -------
    go.Figure
        An interactive Plotly horizontal bar chart of the top N suppliers.
    """

    # First sort descending to correctly identify the TRUE top N by spend,
    # then take only those rows.
    top_df = supplier_summary_df.sort_values(by=spend_column, ascending=False).head(top_n)

    # Then re-sort ascending purely for DISPLAY purposes, so (as explained
    # in chart #2) the single largest supplier visually renders at the top
    # of the horizontal bar chart.
    top_df = top_df.sort_values(by=spend_column, ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_df[spend_column],
            y=top_df[supplier_column],
            orientation="h",
            marker={"color": HEALTHCARE_COLOR_PALETTE[2]},
            hovertemplate="<b>%{y}</b><br>Total Spend: %{x:$,.2f}<extra></extra>",
        )
    )

    return _apply_healthcare_theme(
        fig, title=f"Top {top_n} Suppliers by Spend", x_title="Total Spend ($)", y_title=""
    )


# ---------------------------------------------------------------------------
# CHART 7: monthly_spending_by_supplier_stacked_bar_chart
# ---------------------------------------------------------------------------
def monthly_spending_by_supplier_stacked_bar_chart(
    cleaned_df: pd.DataFrame,
    amount_column: str,
    supplier_column: str,
    year_column: str = "Year",
    month_column: str = "Month",
    top_n_suppliers: int = 6,
) -> go.Figure:
    """
    Build a stacked bar chart showing monthly spend broken down by
    supplier, so each month's total bar is divided into colored segments
    representing how much of that month's spend went to each vendor.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    The monthly line chart (#1) shows THAT spend changed month to month,
    but not WHY. This chart answers that follow-up question directly: if
    March had an unusual spike, this chart shows whether it was driven by
    one specific vendor (e.g. a large one-time equipment purchase from
    Medtronic) or spread evenly across many vendors (e.g. general
    inflation across the board). This is one of the most powerful charts
    for turning "spend went up" into "spend went up BECAUSE of X."

    IMPORTANT DESIGN DECISION -- why we group into "Top N + Other":
    A hospital may have 30-50+ distinct suppliers. Stacking 50 colors into
    one bar chart would be unreadable confetti, and 50 legend entries
    would be useless. Instead, this function keeps the top_n_suppliers
    (by TOTAL spend across the whole dataset) as their own named,
    colored segments, and combines every remaining smaller supplier into
    a single "Other Suppliers" segment. This keeps the chart legible while
    still preserving the correct overall monthly total.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        The cleaned, row-level dataset (NOT a pre-aggregated summary --
        this chart needs to group by month AND supplier together, which
        isn't one of the standard analyzeData.py summary tables).
    amount_column : str
        Name of the numeric purchase amount column.
    supplier_column : str
        Name of the supplier/vendor name column.
    year_column, month_column : str
        Names of the Year/Month columns (from cleanData.py).
    top_n_suppliers : int, default 6
        How many of the biggest suppliers get their own individual color
        segment; everyone else is grouped into "Other Suppliers."

    Returns
    -------
    go.Figure
        An interactive Plotly stacked bar chart.
    """

    df = cleaned_df.copy()

    # Step A: identify the top N suppliers by their TOTAL spend across
    # the entire dataset (not just within a single month), so the same
    # set of "important" suppliers is used consistently across all
    # months in the chart.
    top_suppliers = (
        df.groupby(supplier_column)[amount_column]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_suppliers)
        .index  # .index here gives us just the supplier NAMES, not their totals
    )

    # Step B: create a new column that keeps top-supplier names as-is,
    # but relabels every other supplier as "Other Suppliers". We use
    # pandas' .where(), which keeps values where the condition is True
    # and replaces them elsewhere.
    df["Supplier Group"] = df[supplier_column].where(
        df[supplier_column].isin(top_suppliers), other="Other Suppliers"
    )

    # Step C: build a human-readable "Jan 2024" style period label, exactly
    # as in the monthly line chart, so the x-axis is easy to read.
    df["Period"] = (
        pd.to_datetime(df[year_column].astype(str) + "-" + df[month_column].astype(str) + "-01")
        .dt.strftime("%b %Y")
    )
    # Also keep a true sortable date so we can order the x-axis
    # chronologically (text like "Jan 2024" alone would sort alphabetically,
    # which is wrong -- "Apr" would come before "Jan").
    df["Period Sort Key"] = pd.to_datetime(
        df[year_column].astype(str) + "-" + df[month_column].astype(str) + "-01"
    )

    # Step D: aggregate total spend per Period + Supplier Group -- this is
    # the actual data the stacked bars will be built from.
    grouped = (
        df.groupby(["Period", "Period Sort Key", "Supplier Group"])[amount_column]
        .sum()
        .reset_index()
        .sort_values(by="Period Sort Key")
    )

    fig = go.Figure()

    # Step E: add one bar TRACE per supplier group. Plotly stacks multiple
    # bar traces on top of each other automatically when the layout's
    # barmode is set to "stack" (done below in update_layout). We loop
    # over each unique supplier group and add its own trace/segment.
    unique_groups = grouped["Supplier Group"].unique()
    for i, group_name in enumerate(unique_groups):
        group_data = grouped[grouped["Supplier Group"] == group_name]
        fig.add_trace(
            go.Bar(
                x=group_data["Period"],
                y=group_data[amount_column],
                name=group_name,
                # Cycle through our color palette using modulo (%) so we
                # never run out of colors, even with many supplier groups.
                marker={"color": HEALTHCARE_COLOR_PALETTE[i % len(HEALTHCARE_COLOR_PALETTE)]},
                hovertemplate=f"<b>{group_name}</b><br>%{{x}}<br>Spend: %{{y:$,.2f}}<extra></extra>",
            )
        )

    fig.update_layout(barmode="stack")

    return _apply_healthcare_theme(
        fig, title="Monthly Spending by Supplier", x_title="Month", y_title="Total Spend ($)"
    )


# ---------------------------------------------------------------------------
# CHART 8: purchase_amount_distribution_histogram
# ---------------------------------------------------------------------------
def purchase_amount_distribution_histogram(
    cleaned_df: pd.DataFrame,
    amount_column: str,
    num_bins: int = 40,
) -> go.Figure:
    """
    Build a histogram showing the distribution (spread/shape) of
    individual purchase order amounts.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    Every other chart so far shows TOTALS or trends, but this one answers
    a different question: "what does a NORMAL purchase look like for us,
    and how much variation is there?" Most purchase amounts will cluster
    in a typical range (e.g. most POs are $500-$5,000), with a long
    "tail" of rare, much larger purchases (e.g. capital equipment). This
    chart makes that pattern visually obvious, and is the visual
    foundation for outlier detection (outlierDetect.py) -- an "outlier"
    is, by definition, a purchase far out in that tail, away from where
    most of the data is clustered.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        The cleaned, row-level dataset.
    amount_column : str
        Name of the numeric purchase amount column.
    num_bins : int, default 40
        How many bins (bars) to divide the amount range into. More bins
        show finer detail; fewer bins show a smoother overall shape.

    Returns
    -------
    go.Figure
        An interactive Plotly histogram.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=cleaned_df[amount_column],
            nbinsx=num_bins,
            marker={"color": HEALTHCARE_COLOR_PALETTE[0], "line": {"color": "white", "width": 1}},
            hovertemplate="Range: %{x}<br>Number of Purchases: %{y}<extra></extra>",
        )
    )

    return _apply_healthcare_theme(
        fig,
        title="Distribution of Individual Purchase Amounts",
        x_title="Purchase Amount ($)",
        y_title="Number of Purchase Orders",
    )


# ---------------------------------------------------------------------------
# CHART 9: outlier_scatter_plot
# ---------------------------------------------------------------------------
def outlier_scatter_plot(
    cleaned_df: pd.DataFrame,
    date_column: str,
    amount_column: str,
    outlier_column: Optional[str] = None,
) -> go.Figure:
    """
    Build a scatter plot with every individual purchase order plotted by
    its date (x-axis) and amount (y-axis), with outliers highlighted in a
    distinct color.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    While the histogram (#8) shows the overall SHAPE of spending, this
    chart shows WHICH SPECIFIC purchases are unusual, and WHEN they
    happened. This is the most actionable chart for manual review: a
    Clinical Engineering manager can see red-highlighted points and
    immediately investigate "was this $85,000 purchase in June actually
    authorized/legitimate, or a data entry error, or worth negotiating
    differently next time?" Plotting over time (rather than just listing
    outliers in a table) also reveals whether outliers are clustered in
    a specific period (e.g. all near fiscal year-end) or scattered
    randomly throughout the year.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        The cleaned, row-level dataset.
    date_column : str
        Name of the datetime column (x-axis).
    amount_column : str
        Name of the numeric purchase amount column (y-axis).
    outlier_column : str, optional
        Name of a boolean column (True = this row is an outlier),
        typically produced by outlierDetect.py. If not provided (None),
        every point is plotted in a single neutral color -- this keeps
        this chart usable even before outlier detection has been run.

    Returns
    -------
    go.Figure
        An interactive Plotly scatter plot.
    """

    df = cleaned_df.copy()

    fig = go.Figure()

    if outlier_column is not None and outlier_column in df.columns:
        # Split the data into two groups so each can be drawn as its own
        # trace with its own distinct color and legend entry: normal
        # purchases, and flagged outliers.
        normal_points = df[df[outlier_column] == False]  # noqa: E712 (explicit True/False comparison is clearer here)
        outlier_points = df[df[outlier_column] == True]  # noqa: E712

        fig.add_trace(
            go.Scatter(
                x=normal_points[date_column],
                y=normal_points[amount_column],
                mode="markers",
                name="Normal Purchase",
                marker={"color": NORMAL_POINT_COLOR, "size": 7, "opacity": 0.7},
                hovertemplate="<b>%{x}</b><br>Amount: %{y:$,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=outlier_points[date_column],
                y=outlier_points[amount_column],
                mode="markers",
                name="Outlier",
                marker={
                    "color": OUTLIER_COLOR,
                    "size": 11,
                    "symbol": "diamond",  # a different SHAPE (not just color)
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="<b>%{x}</b><br>Amount: %{y:$,.2f}<br><b>Flagged as Outlier</b><extra></extra>",
            )
        )
    else:
        # No outlier column was supplied -- plot everything as one group
        # in a neutral color so this chart still works standalone.
        fig.add_trace(
            go.Scatter(
                x=df[date_column],
                y=df[amount_column],
                mode="markers",
                name="Purchase Order",
                marker={"color": NORMAL_POINT_COLOR, "size": 7, "opacity": 0.7},
                hovertemplate="<b>%{x}</b><br>Amount: %{y:$,.2f}<extra></extra>",
            )
        )

    return _apply_healthcare_theme(
        fig, title="Purchase Orders Over Time (Outliers Highlighted)", x_title="Date", y_title="Purchase Amount ($)"
    )


# ---------------------------------------------------------------------------
# CHART 10: spending_trend_dashboard
# ---------------------------------------------------------------------------
def spending_trend_dashboard(
    monthly_summary_df: pd.DataFrame,
    supplier_summary_df: pd.DataFrame,
    market_summary_df: pd.DataFrame,
    cost_center_summary_df: pd.DataFrame,
    supplier_column: str,
    market_column: str,
    cost_center_column: str,
    year_column: str = "Year",
    month_column: str = "Month",
    spend_column: str = "Total Spend",
    top_n_suppliers: int = 8,
) -> go.Figure:
    """
    Build a single combined "dashboard" figure with four panels in a 2x2
    grid: the monthly spending trend, top suppliers, market/category
    share, and cost center spending -- all in one image.

    WHY THIS CHART MATTERS TO CLINICAL ENGINEERING:
    Executives and department directors often want a single, glanceable
    "state of procurement spend" view rather than scrolling through 9
    separate charts one at a time. This dashboard combines the four most
    strategically important views -- trend over time, top vendors,
    category mix, and departmental accountability -- into one screen,
    ideal for a leadership meeting or the top of an executive report.

    Parameters
    ----------
    monthly_summary_df, supplier_summary_df, market_summary_df,
    cost_center_summary_df : pd.DataFrame
        The four summary tables from analyzeData.py.
    supplier_column, market_column, cost_center_column : str
        Column names identifying each respective category within its
        summary table.
    year_column, month_column, spend_column : str
        Shared column names used across summaries.
    top_n_suppliers : int, default 8
        How many suppliers to show in the dashboard's supplier panel
        (kept smaller than the standalone top_suppliers_bar_chart's
        default, since this panel has less space in the 2x2 grid).

    Returns
    -------
    go.Figure
        One combined Plotly figure containing all four panels.
    """

    # make_subplots() builds the 2x2 grid layout. The "specs" parameter
    # tells Plotly what TYPE of chart goes in each grid cell -- pie charts
    # need type "domain" (they don't use x/y axes), while line and bar
    # charts use the standard "xy" type.
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Monthly Spending Trend",
            f"Top {top_n_suppliers} Suppliers by Spend",
            "Spending Share by Market",
            "Spending by Cost Center",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "domain"}, {"type": "xy"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # --- Panel 1 (top-left): Monthly trend line ---
    # We build the standalone chart first, then copy its traces into the
    # dashboard -- this reuses our existing, already-tested chart logic
    # instead of duplicating it.
    monthly_fig = monthly_spending_line_chart(monthly_summary_df, year_column, month_column, spend_column)
    for trace in monthly_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # --- Panel 2 (top-right): Top suppliers bar chart ---
    supplier_fig = top_suppliers_bar_chart(supplier_summary_df, supplier_column, spend_column, top_n=top_n_suppliers)
    for trace in supplier_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # --- Panel 3 (bottom-left): Market share pie chart ---
    market_fig = market_spending_pie_chart(market_summary_df, market_column, spend_column)
    for trace in market_fig.data:
        fig.add_trace(trace, row=2, col=1)

    # --- Panel 4 (bottom-right): Cost center bar chart ---
    cost_center_fig = cost_center_spending_bar_chart(cost_center_summary_df, cost_center_column, spend_column)
    for trace in cost_center_fig.data:
        fig.add_trace(trace, row=2, col=2)

    # Apply the shared theme (colors/fonts/background) to the combined
    # figure, and give the whole dashboard an overall title. We don't
    # reuse individual x/y axis titles here since a compact dashboard
    # favors the subplot titles above for context instead.
    fig = _apply_healthcare_theme(fig, title=" Spending Dashboard")

    # A dashboard with 4 panels usually has too many/overlapping legend
    # entries to show cleanly (e.g. the pie chart's legend and the bar
    # charts' hover context aren't directly comparable) -- hiding it keeps
    # the dashboard visually clean, since each panel's own subplot title
    # and axis labels already provide enough context.
    fig.update_layout(showlegend=False, height=700)

    return fig


# ---------------------------------------------------------------------------
# STANDALONE TEST BLOCK
# ---------------------------------------------------------------------------
# Runs only when you execute this file directly: python src/graph.py
# We reuse the same sample dataset shape as analyzeData.py's test block,
# run it through the real summary functions, then build every chart and
# confirm each one returns a valid go.Figure with data in it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Import the real summary functions so this test proves graph.py works
    # correctly with ACTUAL analyzeData.py output, not a hand-built mock.
    from analyzeData import (
        monthly_spending,
        supplier_spending,
        market_spending,
        cost_center_spending,
        account_spending,
    )

    sample_clean_data = {
        "Supplier": [
            "Medtronic", "Ge Healthcare", "Stryker", "Medtronic", "Philips",
            "Stryker", "Medtronic", "Ge Healthcare", "Philips", "Stryker",
            "Medtronic", "Ge Healthcare", "Boston Scientific", "Abbott", "Zimmer Biomet",
        ],
        "Market": [
            "Patient Monitoring", "Imaging", "Surgical", "Patient Monitoring", "Imaging",
            "Surgical", "Patient Monitoring", "Imaging", "Imaging", "Surgical",
            "Patient Monitoring", "Imaging", "Cardiology", "Cardiology", "Surgical",
        ],
        "Cost Center": [
            "ICU", "Radiology", "OR", "ICU", "Radiology", "OR", "ICU",
            "Radiology", "Radiology", "OR", "ICU", "Radiology", "Cath Lab", "Cath Lab", "OR",
        ],
        "Account": [
            "6100", "6200", "6300", "6100", "6200", "6300", "6100",
            "6200", "6200", "6300", "6100", "6200", "6400", "6400", "6300",
        ],
        "PO Amount Ordered": [
            4500.00, 12800.50, 3200.00, 4600.00, 980.25, 3100.00,
            4550.00, 15000.00, 890.00, 3050.00, 47000.00, 13200.00,
            8200.00, 9100.00, 2700.00,
        ],
        "PO Date": pd.to_datetime([
            "2024-01-05", "2024-01-12", "2024-01-20", "2024-02-10", "2024-02-15",
            "2024-02-22", "2024-03-01", "2024-03-10", "2024-03-18", "2024-04-02",
            "2024-04-09", "2024-04-20", "2024-05-02", "2024-05-14", "2024-05-25",
        ]),
    }
    clean_df = pd.DataFrame(sample_clean_data)
    clean_df["Year"] = clean_df["PO Date"].dt.year
    clean_df["Month"] = clean_df["PO Date"].dt.month

    # A simple mock outlier flag: anything over $30,000 is "flagged" here
    # purely to test outlier_scatter_plot() before outlierDetect.py exists.
    clean_df["Is Outlier"] = clean_df["PO Amount Ordered"] > 30000

    # Build the real summary tables using analyzeData.py.
    monthly_df = monthly_spending(clean_df, "PO Amount Ordered")
    supplier_df = supplier_spending(clean_df, "PO Amount Ordered", "Supplier")
    market_df = market_spending(clean_df, "PO Amount Ordered", "Market")
    cost_center_df = cost_center_spending(clean_df, "PO Amount Ordered", "Cost Center")
    account_df = account_spending(clean_df, "PO Amount Ordered", "Account")

    print("Building all 10 charts...\n")

    charts = {
        "monthly_spending_line_chart": monthly_spending_line_chart(monthly_df),
        "supplier_spending_bar_chart": supplier_spending_bar_chart(supplier_df, "Supplier"),
        "market_spending_pie_chart": market_spending_pie_chart(market_df, "Market"),
        "cost_center_spending_bar_chart": cost_center_spending_bar_chart(cost_center_df, "Cost Center"),
        "account_spending_pie_chart": account_spending_pie_chart(account_df, "Account"),
        "top_suppliers_bar_chart": top_suppliers_bar_chart(supplier_df, "Supplier", top_n=15),
        "monthly_spending_by_supplier_stacked_bar_chart": monthly_spending_by_supplier_stacked_bar_chart(
            clean_df, "PO Amount Ordered", "Supplier", top_n_suppliers=4
        ),
        "purchase_amount_distribution_histogram": purchase_amount_distribution_histogram(
            clean_df, "PO Amount Ordered", num_bins=10
        ),
        "outlier_scatter_plot": outlier_scatter_plot(
            clean_df, "PO Date", "PO Amount Ordered", outlier_column="Is Outlier"
        ),
        "spending_trend_dashboard": spending_trend_dashboard(
            monthly_df, supplier_df, market_df, cost_center_df,
            "Supplier", "Market", "Cost Center", top_n_suppliers=5,
        ),
    }

    # Confirm every chart is a valid Plotly figure with at least one trace
    # of data in it, and save each one as an HTML file into graphs/ so
    # they can be opened and visually inspected in a browser.
    import os
    output_dir = "../graphs"
    os.makedirs(output_dir, exist_ok=True)

    for chart_name, fig in charts.items():
        assert isinstance(fig, go.Figure), f"{chart_name} did not return a go.Figure!"
        assert len(fig.data) > 0, f"{chart_name} has no data traces!"
        output_path = os.path.join(output_dir, f"{chart_name}.html")
        fig.write_html(output_path)
        print(f"OK  {chart_name} -> saved to {output_path}")

    print("\nAll 10 charts built successfully and saved to the graphs/ folder.")

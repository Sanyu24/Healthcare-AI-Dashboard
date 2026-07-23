"""
=============================================================================
 report.py
=============================================================================
STEP 6 OF THE PIPELINE: AI-GENERATED EXECUTIVE SUMMARY

This module is what makes Project Walter an INTELLIGENT assistant rather
than just a chart-generator. It takes the already-computed summary tables
from analyzeData.py and outlierDetect.py, sends them to Anthropic's Claude
API, and asks Claude to write a plain-English executive report covering:

    1. Executive Summary
    2. Key Trends
    3. Top Suppliers
    4. Potential Risks
    5. Interesting Insights
    6. Recommendations

IMPORTANT: This module NEVER sends the raw Excel data to Claude. It only
sends small, already-aggregated SUMMARY TABLES (a handful of rows each --
monthly totals, top suppliers, category totals, flagged outliers, cost
center totals). This keeps the data sent to the API minimal, fast, and
focused only on the numbers that actually matter for the summary --
Claude doesn't need (and shouldn't need) to see every individual purchase
order to write a good executive report.

=============================================================================
 HOW PROMPTS ARE SENT TO CLAUDE (beginner explanation)
=============================================================================
Talking to Claude through the API involves three pieces, all bundled into
one function call: client.messages.create(...)

    1. model        -- which Claude model to use (e.g. "claude-sonnet-5")
    2. system       -- a "role" instruction that applies to the WHOLE
                        conversation. This is where we tell Claude:
                        "You are a healthcare procurement financial
                        analyst. Write in six clearly labeled sections."
                        Think of this as Claude's job description.
    3. messages     -- a list of the actual conversation turns. For a
                        one-shot report like this, this is just a single
                        entry: {"role": "user", "content": "<our prompt>"}.
                        The "content" text is where we embed all five
                        summary tables as readable text.

We send this ONE request and wait for Claude's reply -- there's no need
for multi-turn back-and-forth here, since we're asking for one complete
report in one go.

=============================================================================
 HOW THE RESPONSE IS RECEIVED (beginner explanation)
=============================================================================
The API returns a "Message" object (not just a plain string). Claude's
actual written reply lives inside a list called message.content, where
each item is a "content block." For a simple text reply (no tool use,
no images), this list normally contains exactly ONE block of type "text".
We access the reply text as: message.content[0].text

We loop through all blocks defensively (rather than assuming there's
always exactly one) and join their text together, in case the response
ever contains multiple text blocks.

=============================================================================
 API KEY SECURITY
=============================================================================
The API key is NEVER written directly in this file. It is read from the
environment variable ANTHROPIC_API_KEY at runtime using os.environ.get().
This means the key lives in your operating system / hosting platform's
environment settings, or a local .env file that is NOT committed to
source control -- never inside this Python file itself.

WHAT THIS FILE PROVIDES (public functions):
    1. get_anthropic_client()          -> creates an authenticated API client
    2. dataframe_to_prompt_text()      -> converts a summary DataFrame to text
    3. build_executive_summary_prompt() -> assembles the full prompt
    4. call_claude_api()               -> sends the prompt, handles errors
    5. parse_report_sections()         -> splits Claude's reply into 6 sections
    6. generate_executive_report()     -> orchestrator: does all of the above
    7. save_report_to_file()           -> saves the final report to disk
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# 'os' lets us read the API key from an environment variable, and lets us
# create the reports/ folder and build file paths safely.
import os

# 'time' is used for the exponential backoff retry logic (waiting longer
# between each retry attempt after a rate-limit or connection error).
import time

# 're' (regular expressions) is used in parse_report_sections() to split
# Claude's response text into its six labeled sections.
import re

# pandas is needed because every input to this module is a pandas
# DataFrame (the summary tables from analyzeData.py / outlierDetect.py).
import pandas as pd

# The 'anthropic' package is Anthropic's official Python SDK -- it
# provides the Anthropic client class and all of the specific exception
# types we use below for error handling (RateLimitError, etc.).
import anthropic

# Type hints -- purely for readability, they don't change how the code runs.
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# A CUSTOM EXCEPTION FOR THIS MODULE
# ---------------------------------------------------------------------------
class ReportGenerationError(Exception):
    """
    A custom exception type specifically for problems that happen while
    generating the AI executive report.

    WHY DEFINE A CUSTOM EXCEPTION:
    Rather than letting a raw, low-level error from the 'anthropic'
    library bubble all the way up to the Streamlit UI (main.py), we catch
    those specific errors inside this module and re-raise them as ONE
    consistent, friendly exception type. This means main.py only ever
    needs to write ONE try/except block around report generation, instead
    of needing to know about every possible individual error type the
    Anthropic SDK might raise.
    """
    pass


# ---------------------------------------------------------------------------
# FUNCTION 1: get_anthropic_client
# ---------------------------------------------------------------------------
def get_anthropic_client() -> anthropic.Anthropic:
    """
    Create and return an authenticated Anthropic API client.

    WHERE THE API KEY COMES FROM (IMPORTANT SECURITY NOTE):
    This function NEVER hardcodes an API key anywhere in this file.
    Instead, it reads the key from an environment variable called
    ANTHROPIC_API_KEY using os.environ.get(). This is the standard,
    secure way to handle API credentials:
        - The key lives OUTSIDE the source code (in your terminal
          environment, a .env file, or your hosting platform's secret
          manager), so it's never accidentally committed to version
          control or visible to anyone reading this code.
        - To set it locally before running the app, you would run
          (on Mac/Linux): export ANTHROPIC_API_KEY="your-key-here"
          or (on Windows):  set ANTHROPIC_API_KEY=your-key-here
        - In Streamlit Community Cloud or similar hosting, this would be
          set in the platform's "Secrets" settings instead.

    Returns
    -------
    anthropic.Anthropic
        An authenticated client object, ready to send requests to Claude.

    Raises
    ------
    ReportGenerationError
        If the ANTHROPIC_API_KEY environment variable is not set at all,
        with a clear, actionable message telling the user how to fix it.
    """

    # os.environ.get() looks up an environment variable by name and
    # returns None if it isn't set (rather than crashing), which lets us
    # check for it explicitly and give a friendly error message below.
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ReportGenerationError(
            "No Anthropic API key was found. Please set the ANTHROPIC_API_KEY "
            "environment variable before running this application. "
            "(This key should never be written directly in the source code.)"
        )

    # anthropic.Anthropic(api_key=...) creates the client object we'll use
    # to actually send requests. This object handles all the underlying
    # network communication, authentication headers, retries for certain
    # errors, etc.
    client = anthropic.Anthropic(api_key=api_key)

    return client


# ---------------------------------------------------------------------------
# FUNCTION 2: dataframe_to_prompt_text
# ---------------------------------------------------------------------------
def dataframe_to_prompt_text(df: pd.DataFrame, max_rows: int = 15) -> str:
    """
    Convert a summary DataFrame into a compact, readable block of plain
    text suitable for embedding inside a prompt sent to Claude.

    WHY THIS HELPER EXISTS:
    Every summary table (monthly, supplier, market, cost center, outlier)
    needs to be turned into text before it can be included in a prompt --
    an AI model reads text, not a pandas DataFrame object. Rather than
    repeating this conversion logic five separate times, we do it once
    here and reuse it for every summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Any summary table to convert.
    max_rows : int, default 15
        The maximum number of rows to include. Summary tables are usually
        already small, but this protects against sending an excessively
        large table (e.g. hundreds of suppliers) that would make the
        prompt unnecessarily long and expensive to process.

    Returns
    -------
    str
        A plain-text, readable version of the table, with a note added if
        rows were truncated.
    """

    # If the DataFrame is empty (e.g. a summary was skipped because a
    # column didn't exist in this dataset), return a clear note instead
    # of an empty or confusing block of text.
    if df.empty:
        return "(No data available for this summary.)"

    truncated = len(df) > max_rows

    # .head(max_rows) keeps only the first N rows -- since our summary
    # tables from analyzeData.py are already sorted by importance (e.g.
    # highest spend first), keeping the top rows keeps the most important
    # information.
    display_df = df.head(max_rows)

    # .to_string(index=False) renders the DataFrame as a neatly aligned
    # plain-text table, without the extra pandas row-index numbers (which
    # aren't meaningful to Claude and would just add noise).
    text = display_df.to_string(index=False)

    if truncated:
        text += f"\n(... showing top {max_rows} of {len(df)} total rows ...)"

    return text


# ---------------------------------------------------------------------------
# FUNCTION 3: build_executive_summary_prompt
# ---------------------------------------------------------------------------
def build_executive_summary_prompt(
    monthly_summary_df: pd.DataFrame,
    supplier_summary_df: pd.DataFrame,
    market_summary_df: pd.DataFrame,
    outlier_summary_df: pd.DataFrame,
    cost_center_summary_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Assemble the complete prompt that will be sent to Claude, built ONLY
    from summary tables -- never raw, row-level Excel data.

    Parameters
    ----------
    monthly_summary_df : pd.DataFrame
        Output of analyzeData.py's monthly_spending().
    supplier_summary_df : pd.DataFrame
        Output of analyzeData.py's supplier_spending().
    market_summary_df : pd.DataFrame
        Output of analyzeData.py's market_spending().
    outlier_summary_df : pd.DataFrame
        Output of outlierDetect.py's detect_outliers() -- specifically the
        outlier_df (the second returned DataFrame).
    cost_center_summary_df : pd.DataFrame
        Output of analyzeData.py's cost_center_spending().

    Returns
    -------
    Dict[str, str]
        A dictionary with two keys: "system" (the role/instructions
        prompt) and "user" (the prompt containing the actual data),
        ready to be passed straight into call_claude_api().
    """

    # --- The SYSTEM prompt: Claude's "job description" for this task ---
    # This tells Claude WHO to be (a procurement financial analyst
    # writing for hospital leadership) and exactly HOW to format its
    # answer, so the six required sections come back consistently
    # labeled every time, making them easy to parse afterward in
    # parse_report_sections().
    system_prompt = (
        "You are an experienced healthcare procurement financial analyst "
        "writing an executive report for the Clinical Engineering "
        "department at a hospital. You will be given several small summary "
        "tables of procurement spending data (monthly totals, supplier "
        "totals, market/category totals, cost center totals, and a list of "
        "statistically unusual purchase orders). "
        "Write a clear, professional executive report using PLAIN language "
        "that a hospital director (not a data analyst) can easily "
        "understand. Do not simply repeat numbers from the tables -- "
        "interpret them and explain what they mean for the department. "
        "\n\n"
        "Structure your ENTIRE response using EXACTLY these six section "
        "headers, in this order, each on its own line starting with '## ':\n"
        "## Executive Summary\n"
        "## Key Trends\n"
        "## Top Suppliers\n"
        "## Potential Risks\n"
        "## Interesting Insights\n"
        "## Recommendations\n"
        "Do not add any other section headers, and do not add any text "
        "before the first header or after the last section."
    )

    # --- The USER prompt: the actual data, clearly labeled ---
    # We build this piece by piece so it's easy to read and modify later.
    # Each summary table is converted to text using our reusable helper
    # (Function 2), and clearly labeled so Claude knows what each table
    # represents.
    user_prompt_parts = [
        "Here is the procurement spending data to analyze:\n",

        "### Monthly Spending Summary",
        dataframe_to_prompt_text(monthly_summary_df),
        "",

        "### Supplier Spending Summary",
        dataframe_to_prompt_text(supplier_summary_df),
        "",

        "### Market/Category Spending Summary",
        dataframe_to_prompt_text(market_summary_df),
        "",

        "### Cost Center Spending Summary",
        dataframe_to_prompt_text(cost_center_summary_df),
        "",

        "### Flagged Outlier Purchase Orders (statistically unusual)",
        dataframe_to_prompt_text(outlier_summary_df),
        "",

        "Please write the six-section executive report now, following the "
        "structure described in your instructions.",
    ]

    # "\n".join(...) combines every piece above into one single block of
    # text, with a line break between each part.
    user_prompt = "\n".join(user_prompt_parts)

    return {"system": system_prompt, "user": user_prompt}


# ---------------------------------------------------------------------------
# FUNCTION 4: call_claude_api
# ---------------------------------------------------------------------------
def call_claude_api(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-5",
    max_tokens: int = 2000,
    max_retries: int = 3,
) -> str:
    """
    Send a prompt to the Claude API and return Claude's text response,
    with proper error handling and automatic retries for temporary
    failures.

    HOW THE REQUEST IS SENT:
    client.messages.create() is the single function call that sends
    everything to Claude:
        - model: which Claude model answers the request
        - max_tokens: the maximum length of Claude's reply (a safety cap,
          not a target -- Claude will stop naturally once it's done)
        - system: Claude's role/instructions for this whole request
        - messages: a list of conversation turns -- here just one user
          turn containing our data-filled prompt

    HOW THE RESPONSE IS RECEIVED:
    The function returns a "Message" object. The actual reply text lives
    inside response.content, which is a LIST of content blocks (almost
    always just one, for a plain text answer like this). We loop through
    every block, check that its .type is "text", and join all the text
    together -- this is more defensive than assuming there's always
    exactly one block.

    Parameters
    ----------
    system_prompt : str
        The system/role instructions (see build_executive_summary_prompt).
    user_prompt : str
        The user message containing the actual data (see
        build_executive_summary_prompt).
    model : str, default "claude-sonnet-5"
        Which Claude model to use. Kept as a parameter (not hardcoded)
        so it's easy to swap models later without changing any other code.
    max_tokens : int, default 2000
        Maximum length of Claude's response, in tokens (roughly, pieces
        of words). 2000 is comfortably enough for a six-section report.
    max_retries : int, default 3
        How many times to automatically retry the request if it fails
        due to a TEMPORARY problem (rate limiting or a connection issue),
        using "exponential backoff" -- waiting a little longer before
        each subsequent retry attempt.

    Returns
    -------
    str
        Claude's full text response.

    Raises
    ------
    ReportGenerationError
        If the request fails for any reason (invalid API key, model not
        found, malformed request, or repeated temporary failures after
        all retries are exhausted), with a clear, friendly message
        explaining what went wrong.
    """

    client = get_anthropic_client()

    # This loop lets us try the request multiple times if it fails due to
    # a TEMPORARY issue (like a brief server overload or rate limit),
    # rather than giving up on the very first hiccup.
    for attempt in range(1, max_retries + 1):
        try:
            # THIS is the actual API call. Everything before this point
            # was just preparing the pieces we send here.
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )

            # Extract the reply text from the response's content blocks,
            # as explained in this function's docstring above.
            reply_text_parts = []
            for content_block in response.content:
                if content_block.type == "text":
                    reply_text_parts.append(content_block.text)

            full_reply_text = "".join(reply_text_parts)

            # A quick sanity check: if Claude somehow returned a
            # completely empty response, treat that as a failure rather
            # than silently returning nothing.
            if not full_reply_text.strip():
                raise ReportGenerationError(
                    "Claude returned an empty response. Please try again."
                )

            return full_reply_text

        # --- Handle a WRONG/MISSING API KEY ---
        # This is a permanent problem (retrying won't help), so we raise
        # immediately with a clear, actionable message instead of
        # wasting retries on something that can't fix itself.
        except anthropic.AuthenticationError as error:
            raise ReportGenerationError(
                "Authentication with the Anthropic API failed. Please check "
                "that your ANTHROPIC_API_KEY environment variable contains a "
                f"valid, active API key. Original error: {error}"
            )

        # --- Handle a model name that doesn't exist / isn't available ---
        # Also a permanent problem -- raise immediately.
        except anthropic.NotFoundError as error:
            raise ReportGenerationError(
                f"The requested Claude model '{model}' was not found or is "
                f"not available to this API key. Original error: {error}"
            )

        # --- Handle a malformed request (a bug in our own prompt code) ---
        # Also permanent -- retrying the exact same broken request would
        # just fail again the same way.
        except anthropic.BadRequestError as error:
            raise ReportGenerationError(
                f"The request sent to Claude was invalid. This usually "
                f"indicates a bug in how the prompt was built. "
                f"Original error: {error}"
            )

        # --- Handle rate limiting (TEMPORARY -- worth retrying) ---
        except anthropic.RateLimitError as error:
            if attempt == max_retries:
                raise ReportGenerationError(
                    f"The Anthropic API is currently rate-limiting requests, "
                    f"and all {max_retries} retry attempts were exhausted. "
                    f"Please wait a bit and try again. Original error: {error}"
                )
            # "Exponential backoff": wait longer after each failed attempt
            # (2 seconds, then 4, then 8, ...) rather than retrying
            # instantly, which gives the API time to recover.
            wait_seconds = 2 ** attempt
            print(
                f"Rate limited by the Anthropic API (attempt {attempt}/{max_retries}). "
                f"Waiting {wait_seconds} seconds before retrying..."
            )
            time.sleep(wait_seconds)

        # --- Handle a network connection problem (TEMPORARY -- worth retrying) ---
        except anthropic.APIConnectionError as error:
            if attempt == max_retries:
                raise ReportGenerationError(
                    f"Could not connect to the Anthropic API after "
                    f"{max_retries} attempts. Please check your internet "
                    f"connection. Original error: {error}"
                )
            wait_seconds = 2 ** attempt
            print(
                f"Connection error (attempt {attempt}/{max_retries}). "
                f"Waiting {wait_seconds} seconds before retrying..."
            )
            time.sleep(wait_seconds)

        # --- Handle a server-side error on Anthropic's end (TEMPORARY) ---
        # APIStatusError is the general parent class for any non-2xx
        # response Claude's servers send back. We only retry if it's a
        # 5xx (server-side) error -- a 4xx error (other than the specific
        # ones already handled above) is a client-side mistake that won't
        # be fixed by retrying.
        except anthropic.APIStatusError as error:
            if error.status_code >= 500 and attempt < max_retries:
                wait_seconds = 2 ** attempt
                print(
                    f"Anthropic server error (status {error.status_code}, "
                    f"attempt {attempt}/{max_retries}). Waiting {wait_seconds} "
                    f"seconds before retrying..."
                )
                time.sleep(wait_seconds)
            else:
                raise ReportGenerationError(
                    f"The Anthropic API returned an error (status "
                    f"{error.status_code}). Original error: {error}"
                )

        # --- A final catch-all for any other unexpected error ---
        # This ensures the application never crashes with a raw, confusing
        # low-level traceback -- it always surfaces a clear
        # ReportGenerationError instead.
        except Exception as error:
            raise ReportGenerationError(
                f"An unexpected error occurred while contacting Claude: {error}"
            )

    # This line should never actually be reached (the loop above always
    # either returns a result or raises an exception), but it's included
    # defensively so this function always has an explicit return path.
    raise ReportGenerationError("Failed to get a response from Claude after all retries.")


# ---------------------------------------------------------------------------
# FUNCTION 5: parse_report_sections
# ---------------------------------------------------------------------------
def parse_report_sections(report_text: str) -> Dict[str, str]:
    """
    Split Claude's full report text into the six individual sections, so
    each one can be displayed separately (e.g. in its own Streamlit tab
    or expander) instead of only as one giant block of text.

    HOW THIS WORKS:
    build_executive_summary_prompt() instructed Claude to always use
    headers formatted exactly as "## Section Name" on their own line.
    This function uses a regular expression to find each of those headers
    and capture the text that follows, up until the next header (or the
    end of the text).

    Parameters
    ----------
    report_text : str
        The full raw text returned by call_claude_api().

    Returns
    -------
    Dict[str, str]
        A dictionary mapping each section name to its text content, e.g.:
            {
                "Executive Summary": "...",
                "Key Trends": "...",
                "Top Suppliers": "...",
                "Potential Risks": "...",
                "Interesting Insights": "...",
                "Recommendations": "...",
            }
        If a particular section header wasn't found in the response (e.g.
        Claude didn't perfectly follow the formatting instructions), that
        section's value will be an empty string rather than causing an
        error -- so the rest of the report can still be displayed.
    """

    # The six section names we expect, in the order we asked Claude to
    # use them.
    expected_sections = [
        "Executive Summary",
        "Key Trends",
        "Top Suppliers",
        "Potential Risks",
        "Interesting Insights",
        "Recommendations",
    ]

    # re.split() breaks the text apart every time it finds a match for
    # our pattern. The pattern "^## (.+)$" (with re.MULTILINE so "^" and
    # "$" match the start/end of each individual line, not just the whole
    # text) matches a line starting with "## " followed by the section
    # name. Because we wrap "(.+)" in parentheses, re.split() also KEEPS
    # each matched section name in the resulting list, interleaved with
    # the text between headers.
    pieces = re.split(r"^## (.+)$", report_text, flags=re.MULTILINE)

    # After splitting, 'pieces' looks like:
    # ["", "Executive Summary", "<text>", "Key Trends", "<text>", ...]
    # The first element is anything before the first header (should be
    # empty/whitespace if Claude followed instructions), and after that,
    # section names and their text alternate.
    sections: Dict[str, str] = {name: "" for name in expected_sections}

    # We step through the list two-at-a-time: (section_name, section_text)
    for i in range(1, len(pieces) - 1, 2):
        section_name = pieces[i].strip()
        section_text = pieces[i + 1].strip()
        if section_name in sections:
            sections[section_name] = section_text

    return sections


# ---------------------------------------------------------------------------
# FUNCTION 6: generate_executive_report (the orchestrator)
# ---------------------------------------------------------------------------
def generate_executive_report(
    monthly_summary_df: pd.DataFrame,
    supplier_summary_df: pd.DataFrame,
    market_summary_df: pd.DataFrame,
    outlier_summary_df: pd.DataFrame,
    cost_center_summary_df: pd.DataFrame,
    model: str = "claude-sonnet-5",
) -> Dict[str, Any]:
    """
    Run the FULL AI report generation process: build the prompt, call
    Claude, and parse the result -- all in one function call. This is the
    ONE function main.py should actually call.

    Parameters
    ----------
    monthly_summary_df, supplier_summary_df, market_summary_df,
    outlier_summary_df, cost_center_summary_df : pd.DataFrame
        The summary tables from analyzeData.py / outlierDetect.py.
    model : str, default "claude-sonnet-5"
        Which Claude model to use.

    Returns
    -------
    Dict[str, Any]
        {
            "full_text": <the complete raw report text>,
            "sections": <dictionary of the 6 individual sections, from
                         parse_report_sections()>
        }
    """

    # Step 1: build the system + user prompts from our summary tables.
    prompts = build_executive_summary_prompt(
        monthly_summary_df,
        supplier_summary_df,
        market_summary_df,
        outlier_summary_df,
        cost_center_summary_df,
    )

    # Step 2: send the prompt to Claude and get back the raw text reply.
    full_text = call_claude_api(
        system_prompt=prompts["system"],
        user_prompt=prompts["user"],
        model=model,
    )

    # Step 3: split the raw text into the six individual sections for
    # easy separate display later.
    sections = parse_report_sections(full_text)

    return {"full_text": full_text, "sections": sections}


# ---------------------------------------------------------------------------
# FUNCTION 7: save_report_to_file
# ---------------------------------------------------------------------------
def save_report_to_file(report_text: str, output_path: str = "../reports/executive_summary.md") -> str:
    """
    Save the generated report text to a file on disk, inside the
    project's reports/ folder.

    WHY THIS MATTERS:
    Saving a copy of the report as a Markdown (.md) file means the
    Clinical Engineering team has a permanent, shareable, timestamped
    record of each report -- useful for emailing to leadership, archiving
    for audits, or comparing how the analysis has changed over time.

    Parameters
    ----------
    report_text : str
        The report text to save (typically the "full_text" value from
        generate_executive_report()).
    output_path : str, default "../reports/executive_summary.md"
        Where to save the file. The default assumes this function is
        being run from inside the src/ folder, matching this project's
        folder structure.

    Returns
    -------
    str
        The path the file was actually saved to.
    """

    # os.makedirs(..., exist_ok=True) creates the destination folder if
    # it doesn't already exist, and does nothing (without erroring) if it
    # already does. os.path.dirname() extracts just the folder portion of
    # the given path.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write the text to the file. Using "w" mode creates the file if it
    # doesn't exist, or overwrites it if it does.
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    return output_path


# ---------------------------------------------------------------------------
# STANDALONE TEST BLOCK
# ---------------------------------------------------------------------------
# Runs only when you execute this file directly: python src/report.py
#
# IMPORTANT: This test does NOT hardcode an API key. It reads
# ANTHROPIC_API_KEY from the environment, exactly like the real
# application will. If that variable isn't set, the test below still
# proves that:
#   1. The prompt-building logic works correctly on its own (no API
#      access needed for this part).
#   2. Our error handling correctly and clearly reports the missing key,
#      instead of crashing with a confusing raw error.
# If a real API key IS present in the environment, the test will go
# further and make one real request to Claude.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Build small sample summary tables, shaped like real output from
    # analyzeData.py / outlierDetect.py.
    monthly_df = pd.DataFrame({
        "Year": [2024, 2024, 2024, 2024],
        "Month": [1, 2, 3, 4],
        "Total Spend": [20500.50, 8680.25, 20440.00, 63250.00],
        "Number of Orders": [3, 3, 3, 3],
        "Average Order": [6833.50, 2893.42, 6813.33, 21083.33],
    })

    supplier_df = pd.DataFrame({
        "Supplier": ["Medtronic", "Ge Healthcare", "Stryker", "Philips"],
        "Total Spend": [60650.00, 41000.50, 9350.00, 1870.25],
        "Number of Orders": [4, 3, 3, 2],
        "Average Order": [15162.50, 13666.83, 3116.67, 935.13],
        "Percent of Total": [53.73, 36.33, 8.28, 1.66],
    })

    market_df = pd.DataFrame({
        "Market": ["Patient Monitoring", "Imaging", "Surgical"],
        "Total Spend": [60650.00, 42870.75, 9350.00],
        "Number of Orders": [4, 5, 3],
        "Average Order": [15162.50, 8574.15, 3116.67],
        "Percent of Total": [53.73, 37.98, 8.28],
    })

    cost_center_df = pd.DataFrame({
        "Cost Center": ["ICU", "Radiology", "OR"],
        "Total Spend": [60650.00, 42870.75, 9350.00],
        "Number of Orders": [4, 5, 3],
        "Average Order": [15162.50, 8574.15, 3116.67],
        "Percent of Total": [53.73, 37.98, 8.28],
    })

    outlier_df = pd.DataFrame({
        "PO Number": ["PO-1010"],
        "Supplier": ["Medtronic"],
        "Amount": [185000.00],
        "Market": ["Patient Monitoring"],
        "Cost Center": ["ICU"],
        "Reason": [
            "Amount is 6.1 standard deviations above the average (Z-score method); "
            "Amount ($185,000.00) is far above the typical high end of $16,337.50 (IQR method)"
        ],
    })

    # --- Test 1: prompt building (doesn't require API access) ---
    print("=" * 60)
    print("TEST 1: Building the prompt (no API call yet)")
    print("=" * 60)
    prompts = build_executive_summary_prompt(
        monthly_df, supplier_df, market_df, outlier_df, cost_center_df
    )
    print("System prompt preview:\n")
    print(prompts["system"][:300] + "...\n")
    print("User prompt preview:\n")
    print(prompts["user"][:500] + "...\n")
    assert "Executive Summary" in prompts["system"], "System prompt is missing required section names!"
    assert "Medtronic" in prompts["user"], "User prompt is missing supplier data!"
    print("Prompt building works correctly.\n")

    # --- Test 2: parse_report_sections (doesn't require API access) ---
    print("=" * 60)
    print("TEST 2: Parsing a sample Claude-style response")
    print("=" * 60)
    fake_claude_response = (
        "## Executive Summary\n"
        "Spending grew significantly in April, driven by a large Medtronic order.\n\n"
        "## Key Trends\n"
        "Spend increased month over month, with a sharp rise in April.\n\n"
        "## Top Suppliers\n"
        "Medtronic and GE Healthcare account for the majority of total spend.\n\n"
        "## Potential Risks\n"
        "Heavy reliance on Medtronic creates vendor concentration risk.\n\n"
        "## Interesting Insights\n"
        "Patient Monitoring is the largest spending category.\n\n"
        "## Recommendations\n"
        "Consider negotiating volume discounts with top suppliers.\n"
    )
    parsed = parse_report_sections(fake_claude_response)
    for section_name, section_text in parsed.items():
        print(f"--- {section_name} ---")
        print(section_text)
        print()
    assert all(parsed.values()), "One or more sections failed to parse!"
    print("Section parsing works correctly.\n")

    # --- Test 3: real API call (only if a key is actually available) ---
    print("=" * 60)
    print("TEST 3: Calling the real Claude API")
    print("=" * 60)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            result = generate_executive_report(
                monthly_df, supplier_df, market_df, outlier_df, cost_center_df
            )
            print("Received a real response from Claude:\n")
            print(result["full_text"])
            saved_path = save_report_to_file(result["full_text"], "../reports/executive_summary.md")
            print(f"\nReport saved to: {saved_path}")
        except ReportGenerationError as error:
            print(f"Report generation failed with a handled error: {error}")
    else:
        print(
            "No ANTHROPIC_API_KEY environment variable was found in this "
            "environment, so the real API call is being skipped.\n"
            "Confirming our error handling still behaves correctly instead:"
        )
        try:
            get_anthropic_client()
        except ReportGenerationError as error:
            print(f"\nCorrectly caught missing API key with a clear message:\n  {error}")

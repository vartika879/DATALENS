"""DataLens Streamlit UI.

Thin layer over `datalens.pipeline`. Keep UI logic here; business logic
lives in `src/datalens/`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from datalens.pipeline import run_pipeline
from datalens.schema import SchemaExtractionError, extract_schema

st.set_page_config(page_title="DataLens", page_icon="🔎", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# --- Session state ----------------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None


# --- Helpers ----------------------------------------------------------------


def _save_upload(uploaded_file) -> Path:
    """Save an uploaded file to data/ with a timestamped name."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = Path(uploaded_file.name).name
    path = DATA_DIR / f"uploaded_{timestamp}_{safe_name}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _render_scalar(execution) -> None:
    st.markdown("### Answer")
    st.markdown(f"## `{execution.result_data}`")


def _render_dataframe(execution) -> None:
    st.markdown("### Answer")
    data = execution.result_data
    rows = data["rows"]
    if not rows:
        st.info("No rows in the result.")
        return
    result_df = pd.DataFrame(rows, columns=data["columns"])
    st.dataframe(result_df, use_container_width=True)
    if data["truncated"]:
        st.caption(f"Showing {len(rows)} of {data['shape'][0]} rows (truncated for display).")


def _render_series(execution) -> None:
    st.markdown("### Answer")
    data = execution.result_data
    series_df = pd.DataFrame(
        {"value": data["values"]},
    )
    st.dataframe(series_df, use_container_width=True)
    if data["truncated"]:
        st.caption("Result truncated for display.")


def _render_none(execution) -> None:
    st.markdown("### Answer")
    st.info("Code ran successfully but returned no value.")


def _render_result(execution) -> None:
    renderer = {
        "scalar": _render_scalar,
        "dataframe": _render_dataframe,
        "series": _render_series,
        "none": _render_none,
    }
    renderer.get(execution.result_type, _render_none)(execution)


# --- Layout -----------------------------------------------------------------

st.title("🔎 DataLens")
st.caption("Ask questions about your CSV in plain English.")

# Sidebar — upload
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    if uploaded is not None and uploaded.name != st.session_state.dataset_name:
        path = _save_upload(uploaded)
        try:
            schema = extract_schema(path)
        except SchemaExtractionError as exc:
            st.error(f"Could not read CSV: {exc}")
        else:
            st.session_state.df = pd.read_csv(path)
            st.session_state.schema = schema
            st.session_state.dataset_name = uploaded.name
            st.success(f"Loaded {schema.row_count} rows × {schema.column_count} columns")

    if st.session_state.schema is not None:
        with st.expander("Schema"):
            st.write(f"**File:** {st.session_state.schema.file_name}")
            st.write(f"**Rows:** {st.session_state.schema.row_count}")
            for col in st.session_state.schema.columns:
                st.write(
                    f"- `{col.name}` ({col.dtype}) — "
                    f"{col.null_count} nulls ({col.null_percentage}%)"
                )

# Main area
question = st.text_input(
    "Your question",
    placeholder="e.g. What is the average salary?",
    disabled=st.session_state.df is None,
)

ask = st.button("Ask", type="primary", disabled=st.session_state.df is None)

if ask and question.strip():
    with st.spinner("Thinking..."):
        result = run_pipeline(
            schema=st.session_state.schema,
            question=question.strip(),
            df=st.session_state.df,
        )

    if result.status == "llm_error":
        st.error(f"LLM error: {result.error_message}")
    elif result.status == "error":
        st.error(f"Execution failed: {result.error_message}")
        if result.generated:
            with st.expander("Generated code (failed)"):
                st.code(result.generated.code, language="python")
            if result.execution and result.execution.error_traceback:
                with st.expander("Traceback"):
                    st.code(result.execution.error_traceback, language="text")
    else:
        _render_result(result.execution)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total latency", f"{result.total_duration_ms:.0f} ms")
        col2.metric(
            "Code execution",
            f"{result.execution.duration_ms:.0f} ms",
        )
        col3.metric(
            "Model estimate",
            f"{result.generated.self_reported_confidence:.2f}",
            help="Self-reported by the LLM. Not a calibrated probability.",
        )
        with st.expander("Generated code"):
            st.code(result.generated.code, language="python")
            st.caption(result.generated.explanation)

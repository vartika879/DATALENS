"""DataLens Streamlit UI.

All data access goes through the FastAPI backend via DataLensClient.
No business logic lives here — only presentation and orchestration.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.config import get_settings
from datalens.ui.client import DataLensAPIError, DataLensClient

st.set_page_config(
    page_title="DataLens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Session state ----------------------------------------------------------

for key in ("dataset_id", "dataset_name", "session_id"):
    st.session_state.setdefault(key, None)


# --- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    settings = get_settings()
    api_url = st.text_input(
        "API URL",
        value=settings.datalens_api_url,
        help="Base URL of the DataLens FastAPI backend.",
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Server API key. Ask the operator for the value.",
    )

    client: DataLensClient | None = None
    if api_url and api_key:
        client = DataLensClient(base_url=api_url, api_key=api_key)

    st.divider()
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    if (
        uploaded is not None
        and client is not None
        and uploaded.name != st.session_state.dataset_name
    ):
        with st.spinner("Uploading and analyzing schema..."):
            try:
                resp = client.upload(uploaded.name, uploaded.getvalue())
                st.session_state.dataset_id = resp["dataset_id"]
                st.session_state.dataset_name = uploaded.name
                st.session_state.session_id = None  # new dataset → new session
                st.success(f"Loaded {resp['row_count']} rows × {resp['column_count']} columns")
            except DataLensAPIError as exc:
                st.error(f"Upload failed: {exc.detail}")

    if st.session_state.dataset_name:
        st.caption(f"Current dataset: **{st.session_state.dataset_name}**")


# --- Main area --------------------------------------------------------------

st.title("🔎 DataLens")
st.caption("Ask questions about your CSV in plain English.")

if client is None:
    st.info("👈 Enter the **API URL** and **API Key** in the sidebar to get started.")
    st.stop()

if st.session_state.dataset_id is None:
    st.info("👈 **Upload a CSV** from the sidebar to get started.")
    st.stop()

tab_ask, tab_history = st.tabs(["Ask", "History"])


# --- Ask tab ----------------------------------------------------------------

with tab_ask:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the average salary?",
    )
    ask = st.button("Ask", type="primary", disabled=not question.strip())

    if ask and question.strip():
        with st.spinner("Thinking..."):
            try:
                # Ensure session exists
                if st.session_state.session_id is None:
                    sess = client.create_session(st.session_state.dataset_id)
                    st.session_state.session_id = sess["session_id"]

                resp = client.ask(st.session_state.session_id, question.strip())
            except DataLensAPIError as exc:
                st.error(f"Request failed: {exc.detail}")
            else:
                st.markdown("### Answer")
                st.write(resp["answer"])

                col1, col2, col3 = st.columns(3)
                col1.metric("Status", resp["status"])
                col2.metric("Latency", f"{resp['latency_ms']:.0f} ms")
                col3.metric("Retries", resp["retry_count"])

                if resp.get("code"):
                    with st.expander("Generated code"):
                        st.code(resp["code"], language="python")
                if resp.get("explanation"):
                    st.caption(resp["explanation"])

                # Inline feedback buttons
                st.markdown("**Was this answer helpful?**")
                fb_col1, fb_col2 = st.columns(2)
                if fb_col1.button("👍 Yes", key=f"up_{resp['run_id']}"):
                    try:
                        client.feedback(resp["run_id"], "up")
                        st.success("Thanks!")
                    except DataLensAPIError as exc:
                        st.warning(f"Feedback failed: {exc.detail}")
                if fb_col2.button("👎 No", key=f"down_{resp['run_id']}"):
                    try:
                        client.feedback(resp["run_id"], "down")
                        st.success("Thanks — we'll review this.")
                    except DataLensAPIError as exc:
                        st.warning(f"Feedback failed: {exc.detail}")


# --- History tab ------------------------------------------------------------

with tab_history:
    if st.session_state.session_id is None:
        st.info("Ask a question first to build history.")
    else:
        try:
            hist = client.history(st.session_state.session_id)
        except DataLensAPIError as exc:
            st.error(f"History unavailable: {exc.detail}")
        else:
            runs = hist.get("runs", [])
            if not runs:
                st.info("No runs yet in this session.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Time": r["created_at"],
                                "Question": r["question"],
                                "Status": r["status"],
                                "Answer": r["answer"],
                                "Latency (ms)": round(r["latency_ms"], 1),
                            }
                            for r in runs
                        ]
                    ),
                    width="stretch",
                )

"""Optional Streamlit dashboard shell.

Run an experiment first:

    python run_habmab.py --output reports/latest

Then, if Streamlit is installed:

    streamlit run app/streamlit_app.py
"""

from pathlib import Path


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit("Streamlit is optional. Install streamlit to run the dashboard.") from exc

    report = Path("reports/latest/habmab_report.html")
    st.set_page_config(page_title="HABMAB", layout="wide")
    st.title("HABMAB")
    st.caption("Harmful Algal Bloom Multi-Armed Bandit for adaptive coastal monitoring.")
    if not report.exists():
        st.warning("Run `python run_habmab.py --output reports/latest` first.")
        return
    st.components.v1.html(report.read_text(encoding="utf-8"), height=1200, scrolling=True)


if __name__ == "__main__":
    main()

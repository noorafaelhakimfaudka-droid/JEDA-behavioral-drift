import pathlib
import streamlit as st
import streamlit.components.v1 as components

from helpers import load_models, score_user_timeline

ROOT = pathlib.Path(__file__).resolve().parent
_jeda_component = components.declare_component(
    "jeda_ui",
    path=str(ROOT / "artifacts/ui_component"),
)


@st.cache_resource
def _warm_models():
    return load_models()


def main():
    st.set_page_config(
        page_title="Jeda - Pemantauan Ritme Harian",
        page_icon="J",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _warm_models()

    st.markdown(
        """
        <style>
        #MainMenu, footer, header[data-testid="stHeader"] {display: none;}
        .stApp {background-color: #F8F5F2;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        iframe[title*="jeda_ui"], .stCustomComponentV1 {
            border: none;
            width: 100% !important;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    scored = st.session_state.get("jeda_scored_result")
    last_req_id = st.session_state.get("jeda_last_req_id", 0)

    comp_event = _jeda_component(
        scored_result=scored,
        key="jeda_main_ui",
        height=1180,
    )

    if isinstance(comp_event, dict) and "entries" in comp_event:
        req_id = int(comp_event.get("req_id", 0))
        if req_id != last_req_id:
            st.session_state["jeda_last_req_id"] = req_id
            entries = comp_event.get("entries") or []
            if len(entries) == 0:
                st.session_state["jeda_scored_result"] = None
            else:
                res = score_user_timeline(
                    entries,
                    comp_event.get("profile") or {},
                )
                res["req_id"] = req_id
                res["sig"] = str(comp_event.get("sig") or "")
                st.session_state["jeda_scored_result"] = res
            st.rerun()


if __name__ == "__main__":
    main()

"""Local Streamlit client; authorization and business decisions stay on the API."""

from uuid import uuid4

import httpx
import streamlit as st

st.set_page_config(page_title="SupportPilot", page_icon="💬")
st.title("SupportPilot · local synthetic demo")
st.caption("RU · KK · EN · RU–KK. Research prototype; semantic quality NOT_EVALUATED.")
token = st.sidebar.text_input("Customer bearer token", type="password")
language = st.sidebar.selectbox("Reply language / Язык / Тіл", ["auto", "ru", "kk", "en"])
confirm = st.sidebar.checkbox("Confirm cancellation request", value=False)
if st.session_state.get("active_token") != token:
    st.session_state.clear()
    st.session_state.active_token = token
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = None
if st.sidebar.button("New session"):
    st.session_state.messages = []
    st.session_state.session_id = None
    st.rerun()

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.text(item["content"])
        if item.get("sources"):
            with st.expander("Retrieved sources (relevance does not prove an answer)"):
                for source in item["sources"]:
                    st.text(f"{source['doc_id']} · {source['chunk_id']} · {source['locator']}")
                    st.text(source["text"])
        if item.get("diagnostic"):
            st.caption(item["diagnostic"])

if text := st.chat_input("Message / Сообщение / Хабарлама", disabled=not bool(token)):
    body = {
        "message": text,
        "confirm_cancellation": confirm,
        "request_id": str(uuid4()),
        "session_id": st.session_state.session_id,
        "language": None if language == "auto" else language,
    }
    try:
        with httpx.Client(
            base_url="http://127.0.0.1:8000",
            timeout=185,
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            reply = client.post("/chat", json=body)
            reply.raise_for_status()
            result = reply.json()
        st.session_state.session_id = result["session_id"]
        st.session_state.messages.extend(
            [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": result["response"]["answer"],
                    "sources": result["sources"],
                    "diagnostic": (
                        f"{result['delivery']} · {result['diagnostic']} · "
                        f"ticket={result['escalation_id']}"
                    ),
                },
            ]
        )
        st.session_state.messages = st.session_state.messages[-40:]
        st.rerun()
    except httpx.HTTPError:
        st.error("API unavailable or request rejected. No delivery is confirmed.")

if token and st.sidebar.button("Refresh my requests"):
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(
                "http://127.0.0.1:8000/escalations", headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
        for ticket in response.json():
            st.sidebar.text(f"{ticket['escalation_id']} · {ticket['status']}")
    except httpx.HTTPError:
        st.sidebar.error("Queue status unavailable")

# app.py

import streamlit as st
import json
from engine import GuardianEngine

# ---------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLES
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Guardian AI — Moderation Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, modern UI styling adjustments + overflow fixes for long text
st.markdown("""
<style>
    /* Compact, rounded borders for chat input */
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Ensure markdown and container text wraps properly without truncating */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Engine in session state so it persists across renders
if "engine" not in st.session_state:
    st.session_state.engine = GuardianEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_audit" not in st.session_state:
    st.session_state.latest_audit = None

# Helper to process message and update session audit
def process_message(sender: str, text: str):
    if not text.strip():
        return
    
    # Run audit through GuardianEngine
    audit_result = st.session_state.engine.audit_message(sender, text)
    
    # Update local UI session history
    st.session_state.chat_history.append({
        "sender": sender,
        "text": text,
        "audit": audit_result
    })
    st.session_state.latest_audit = audit_result

# ---------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS & PRESET SCENARIOS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Guardian AI")
    st.caption("Real-Time Gaming Moderation Engine")
    st.divider()

    st.markdown("##### ⚡ Quick Scenarios")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button(" Clean Banter", use_container_width=True):
            st.session_state.engine.reset_chat()
            st.session_state.chat_history = []
            process_message("PlayerOne", "Nice shot mate!")
            process_message("GamerBoy", "Too easy, diff gap in mid lane.")

        if st.button(" Leetspeak", use_container_width=True):
            st.session_state.engine.reset_chat()
            st.session_state.chat_history = []
            process_message("SaltyPlayer", "you trash zero skill")
            process_message("SaltyPlayer", "g0 k1ll ur53lf un1st4ll l1f3")

    with col_s2:
        if st.button(" Harassment", use_container_width=True):
            st.session_state.engine.reset_chat()
            st.session_state.chat_history = []
            process_message("GamerGirl99", "I'll cover point B guys.")
            process_message("ToxikUser", "Is that a girl on mic? Go back to the kitchen and make me a sandwich.")

        if st.button(" Child Safety", use_container_width=True):
            st.session_state.engine.reset_chat()
            st.session_state.chat_history = []
            process_message("MiniGamer", "im only 10 years old this game is hard")
            process_message("PredatorX", "cool what school do you go to? add me on snapchat right now")

    st.divider()
    if st.button("🔴 Reset Lobby", type="primary", use_container_width=True):
        st.session_state.engine.reset_chat()
        st.session_state.chat_history = []
        st.session_state.latest_audit = None
        st.rerun()

# ---------------------------------------------------------------------------
# 3. MAIN DASHBOARD LAYOUT
# ---------------------------------------------------------------------------
st.title("🎮 Live Lobby Moderation")
st.caption("Context-Aware RAG Engine powered by **ChromaDB** & **Gemini 3.5 Flash**")

col_left, col_right = st.columns([1.1, 1], gap="medium")

# --- LEFT COLUMN: LIVE CHAT STREAM ---
with col_left:
    st.subheader("💬 Chat Feed")
    
    # Modern, compact height chat container
    chat_container = st.container(height=320)
    with chat_container:
        if not st.session_state.chat_history:
            st.caption("✨ Lobby is quiet. Select a scenario or type below.")
        else:
            for msg in st.session_state.chat_history:
                audit = msg["audit"]
                is_flagged = audit.get("is_flagged", False)
                severity = audit.get("severity", "NONE")
                
                badge = f"🚨 {severity}" if is_flagged else f"✅ {severity}"
                
                with st.chat_message("user" if not is_flagged else "assistant"):
                    st.markdown(f"**{msg['sender']}**: {msg['text']}")
                    st.caption(f"Status: `{badge}`")

    # Compact Input Form
    with st.form("send_message_form", clear_on_submit=True):
        input_col1, input_col2 = st.columns([1, 3])
        with input_col1:
            sender_input = st.text_input("Sender", value="Player_1", label_visibility="collapsed", placeholder="Username")
        with input_col2:
            text_input = st.text_input("Message", value="", label_visibility="collapsed", placeholder="Type chat message...")
        
        submit_button = st.form_submit_button("Send & Audit", use_container_width=True)
        if submit_button and text_input:
            process_message(sender_input, text_input)
            st.rerun()

# --- RIGHT COLUMN: AI SAFETY AUDIT & RAG INSPECTOR ---
with col_right:
    st.subheader("📊 Audit & RAG Analysis")
    
    audit = st.session_state.latest_audit
    if not audit:
        st.info("Send a message or pick a preset scenario to view real-time audit details.")
    else:
        is_flagged = audit.get("is_flagged", False)
        severity = audit.get("severity", "NONE")
        
        # Header Status Banner
        if not is_flagged:
            st.success(f"**CLEAN PLAY** — Severity: {severity}")
        else:
            if severity in ["CRITICAL", "HIGH"]:
                st.error(f"**VIOLATION DETECTED** — Severity: {severity}")
            else:
                st.warning(f"**POTENTIAL VIOLATION** — Severity: {severity}")

        # Metrics Card Container (Swapped st.metric for Markdown to prevent '...' ellipsis)
        with st.container(border=True):
            m1, m2 = st.columns(2)
            with m1:
                st.caption("Action Recommended")
                st.markdown(f"**{audit.get('recommended_action', 'N/A')}**")
            with m2:
                st.caption("Target Demographic")
                st.markdown(f"**{audit.get('target_group', 'N/A')}**")

            st.divider()
            st.markdown(f"**Violated Policy:** `{audit.get('violated_rule', 'None')}`")
            st.markdown(f"**Reasoning:** {audit.get('summary_reasoning', 'N/A')}")

        # RAG Inspector Box
        with st.expander("🔍 ChromaDB Vector Match", expanded=True):
            rule_id = audit.get('retrieved_rule_id', 'NONE')
            category = audit.get('retrieved_rule_category', 'Standard Clean Conversation')
            
            if not is_flagged or rule_id == "NONE":
                st.write("**Rule ID:** `NONE (Clean Play)`")
                st.write("**Category:** Standard Clean Conversation")
            else:
                st.write(f"**Rule ID:** `{rule_id}`")
                st.write(f"**Category:** {category}")
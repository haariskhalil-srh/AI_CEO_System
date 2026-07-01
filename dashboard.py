import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from datetime import datetime

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
# FIX: Sidebar is now expanded by default so the CEO sees the agent
st.set_page_config(
    page_title="NVIDIA Strategic Intelligence",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] {
        background-color: #1a1c24; border-left: 5px solid #76b900; padding: 15px; border-radius: 4px;
    }
    h1, h2, h3, h4 { color: #76b900 !important; font-family: 'Helvetica Neue', sans-serif; }
    .stProgress > div > div > div > div { background-color: #76b900; }
    .section-divider { border-bottom: 2px solid #333; margin-top: 30px; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60) 
def load_report():
    file_path = "ceo_intelligence_report.json"
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def create_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = int(value),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'color': '#fafafa'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#fafafa'},
            'bar': {'color': "#76b900"}, 
            'bgcolor': "#1a1c24",
            'steps' : [{'range': [0, 40], 'color': "#331414"}, {'range': [40, 60], 'color': "#2b2b2b"}, {'range': [60, 100], 'color': "#1a3300"}], 
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#fafafa"})
    return fig

def parse_confidence(val):
    try:
        if isinstance(val, str):
            val = float(val.replace('%', '').strip())
        val = float(val)
        if 0 < val <= 1.0:
            return int(val * 100)
        return int(val)
    except:
        return 0

report_data = load_report()

if not report_data:
    st.error("Intelligence report not found or corrupted. Please execute agent.py to generate the JSON artifact.")
    st.stop()

# ==========================================
# COPILOT SIDEBAR: Ask the Agent
# ==========================================
# This isolates the chat UI and perfectly pins the input box to the bottom of the sidebar
with st.sidebar:
    st.header("Agent Copilot")
    st.markdown("Your autonomous strategic advisor.")
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello CEO. I am your intelligence agent. How can I help you today?"}]
        st.rerun()
        
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello CEO. I am your intelligence agent. How can I help you today?"}]

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input natively pins to the bottom of the sidebar
    if prompt := st.chat_input("Ask the agent"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # IPC FIX: Pass a sliding memory window (last 5 messages) as JSON, not just text
        memory_payload = st.session_state.messages[-5:]
        with open("pending_query.json", "w", encoding="utf-8") as f:
            json.dump(memory_payload, f)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context"):
                timeout = 120
                start_time = time.time()
                
                # Wait for worker to delete the JSON query file
                while os.path.exists("pending_query.json"):
                    time.sleep(1)
                    if time.time() - start_time > timeout:
                        st.error("Inference timed out. Ensure worker.py is running.")
                        st.stop()
                
                if os.path.exists("agent_response.txt"):
                    with open("agent_response.txt", "r", encoding="utf-8") as f:
                        answer = f.read()
                    os.remove("agent_response.txt")
                else:
                    answer = "Error: Worker finished but response file was lost."
                    
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


# ==========================================
# MAIN DASHBOARD AREA (Sections 1-7)
# ==========================================
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg", width=100)
with col2:
    st.title("Executive Strategic Intelligence")

# ==========================================
# SECTION 1: Company Overview
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("Section 1: Company Overview")
overview = report_data.get("company_overview", {})
c1, c2, c3 = st.columns(3)

c1.metric("Docs Collected", overview.get("total_documents", 0))
c2.metric("Data Sources", overview.get("data_sources_count", 0))

raw_date = overview.get("last_update", "N/A")
try:
    dt = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S UTC")
    formatted_date = dt.strftime("%B %d, %Y")
except:
    formatted_date = raw_date
c3.metric("Last Update", formatted_date)

# ==========================================
# SECTION 2: Market Intelligence
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("Section 2: Market Intelligence")
market_intel = report_data.get("market_intelligence", {})

with st.container(border=True):
    st.subheader("Recent News")
    for item in market_intel.get("recent_news", []): 
        st.write(f"• {item}")

with st.container(border=True):
    st.subheader("Competitor Activities")
    for item in market_intel.get("competitor_activities", []): 
        st.write(f"• {item}")

with st.container(border=True):
    st.subheader("Emerging Technologies")
    for item in market_intel.get("emerging_technologies", []): 
        st.write(f"• {item}")

with st.container(border=True):
    st.subheader("Important Announcements")
    for item in market_intel.get("important_announcements", []): 
        st.write(f"• {item}")

# ==========================================
# SECTION 3: Opportunity Monitor
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("Section 3: Opportunity Monitor")
for opp in report_data.get("opportunities", []):
    with st.container(border=True):
        st.markdown(f"#### {opp.get('title', 'Unknown Opportunity')}")
        st.write(f"**Impact Level:** {opp.get('impact_level', 'N/A')}")
        st.write(f"**Evidence:** {opp.get('verbatim_quote_from_context', 'N/A')}")
        conf = parse_confidence(opp.get('confidence_score', 0))
        st.progress(conf / 100.0, text=f"{conf}% Confidence")

# ==========================================
# SECTION 4: Risk Monitor
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("Section 4: Risk Monitor")
for risk in report_data.get("risks", []):
    with st.container(border=True):
        st.markdown(f"#### {risk.get('title', 'Unknown Risk')}")
        st.write(f"**Category:** {risk.get('category', 'N/A')} | **Severity:** {risk.get('severity_level', 'N/A')}")
        st.write(f"**Evidence:** {risk.get('verbatim_quote_from_context', 'N/A')}")
        conf = parse_confidence(risk.get('confidence_score', 0))
        st.progress(conf / 100.0, text=f"{conf}% Confidence")

# ==========================================
# SECTION 5: Sentiment Analysis
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("Section 5: Sentiment Analysis")
sentiment = report_data.get("sentiment_analysis", {})

justification = sentiment.get("sentiment_justification", "")
if justification:
    st.info(f"**Market Consensus:** {justification}")

s_col1, s_col2, s_col3 = st.columns([1, 1, 2])
with s_col1:
    st.plotly_chart(create_gauge(sentiment.get("news_sentiment_score", 50), "News Sentiment"), use_container_width=True)
with s_col2:
    st.plotly_chart(create_gauge(sentiment.get("public_sentiment_score", 50), "Public Sentiment"), use_container_width=True)
with s_col3:
    trend_data = sentiment.get("historical_trend", [])
    if trend_data:
        df = pd.DataFrame(trend_data)
        fig = px.line(df, x="date", y="score", title="Sentiment Trends")
        fig.update_traces(line_color="#76b900", line_width=3, marker=dict(size=8))
        fig.update_layout(yaxis=dict(range=[0, 100]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#fafafa"})
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# SECTION 6: Strategic Recommendations
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("Section 6: Strategic Recommendations")
for idx, rec in enumerate(report_data.get("recommendations", [])):
    with st.expander(f"Recommendation {idx+1}: {rec.get('recommendation', 'N/A')}"):
        st.markdown(f"### {rec.get('recommendation', 'N/A')}")
        st.write(f"**Priority:** {rec.get('priority', 'N/A')} | **Risk Level:** {rec.get('risk_level', 'N/A')}")
        st.write(f"**Expected Impact:** {rec.get('expected_impact', 'N/A')}")
        st.write(f"**Supporting Evidence:** {rec.get('supporting_evidence', 'N/A')}")
        st.markdown(f"**Source Verification:** [{rec.get('source_url', 'Missing URL')}]({rec.get('source_url', '#')})")

# ==========================================
# SECTION 7: CEO Briefing
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("Section 7: CEO Briefing")
briefing = report_data.get("ceo_briefing", {})
st.info(f"**What happened?**\n\n{briefing.get('what_happened', 'N/A')}")
st.warning(f"**Why does it matter?**\n\n{briefing.get('why_it_matters', 'N/A')}")
st.success(f"**What should management do next?**\n\n{briefing.get('management_next_steps', 'N/A')}")
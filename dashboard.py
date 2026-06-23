import streamlit as st
import json
import pandas as pd
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------
# Page Configuration & CSS Styling
# ---------------------------------------------------------
st.set_page_config(page_title="AI CEO: NVIDIA Intelligence", layout="wide", initial_sidebar_state="collapsed")

# FLUSH-LEFT CSS: Prevents Streamlit's Markdown engine from treating it as a code block
st.markdown("""<style>
.stApp { background-color: #0e0e0e; color: #ffffff; }
h1, h2, h3 { color: #76b900 !important; font-weight: 700 !important; }
.kpi-card { background-color: #1a1a1a; border: 1px solid #333333; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.4); transition: border-color 0.3s ease; height: 100%; display: flex; flex-direction: column; justify-content: center; }
.kpi-card:hover { border-color: #76b900; }
.kpi-title { color: #76b900; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.kpi-value { color: #ffffff; font-size: 1.6rem; font-weight: bold; word-wrap: break-word; }
.info-card { background-color: #1a1a1a; border-left: 5px solid #76b900; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
.info-title { color: #76b900; font-size: 1.2rem; font-weight: bold; margin-bottom: 5px; }
.info-desc { color: #dddddd; font-size: 1rem; line-height: 1.5; }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_clean_date():
    day = datetime.now().day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return datetime.now().strftime(f"{day}{suffix} %B, %Y")

def load_data():
    file_path = "ceo_intelligence_report.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

data = load_data()
if not isinstance(data, dict): data = {} # Absolute failsafe

# ---------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------

st.title("NVIDIA Strategic Intelligence Agent")
st.markdown("Automated Executive Advisory System")
st.divider()

# Section 1: Company Overview
st.header("Section 1: Company Overview")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown("""<div class="kpi-card"><div class="kpi-title">Company Name</div><div class="kpi-value">NVIDIA Corporation</div></div>""", unsafe_allow_html=True)
with c2: st.markdown("""<div class="kpi-card"><div class="kpi-title">Industry</div><div class="kpi-value">Semiconductors & AI</div></div>""", unsafe_allow_html=True)
with c3: st.markdown("""<div class="kpi-card"><div class="kpi-title">Collected Documents</div><div class="kpi-value">120 Documents</div></div>""", unsafe_allow_html=True)
with c4: st.markdown("""<div class="kpi-card"><div class="kpi-title">Data Sources</div><div class="kpi-value">3 Sources</div></div>""", unsafe_allow_html=True)
with c5: st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Last Updated</div><div class="kpi-value">{get_clean_date()}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# Section 2: Market Intelligence
st.header("Section 2: Market Intelligence")
st.markdown("<p style='color:#dddddd; margin-bottom: 20px;'><strong>Emerging Technologies & Industry Trends</strong></p>", unsafe_allow_html=True)
trends = data.get('trends', [])
if isinstance(trends, list):
    for trend in trends:
        if not isinstance(trend, dict): continue
        st.markdown(f"""<div class="info-card">
<div class="info-title">{trend.get('title', 'N/A')}</div>
<div class="info-desc">{trend.get('description', 'N/A')}</div>
</div>""", unsafe_allow_html=True)
st.markdown("<p style='font-style: italic; color: #888888; margin-top: 15px;'>Note: Recent news, competitor activities, and corporate announcements are dynamically factored into the evidence engine below.</p>", unsafe_allow_html=True)
st.divider()

# Section 3: Opportunity Monitor
st.header("Section 3: Opportunity Monitor")
opportunities = data.get('opportunities', [])
if isinstance(opportunities, list):
    for opp in opportunities:
        if not isinstance(opp, dict): continue
        st.markdown(f"""<div class="info-card">
<div class="info-title">{opp.get('title', 'N/A')}</div>
<div style="color: #aaaaaa; font-size: 0.9rem; margin-bottom: 10px;">
<strong>Impact:</strong> {opp.get('impact_level', 'N/A')} &nbsp;|&nbsp; <strong>Confidence:</strong> {opp.get('confidence_score', 'N/A')}%
</div>
<div class="info-desc">{opp.get('evidence', 'N/A')}</div>
</div>""", unsafe_allow_html=True)
st.divider()

# Section 4: Risk Monitor
st.header("Section 4: Risk Monitor")
risks = data.get('risks', [])
if isinstance(risks, list):
    for risk in risks:
        if not isinstance(risk, dict): continue
        sev = str(risk.get('severity_level', 'Medium'))
        s_color = "#ff4b4b" if sev.lower() == 'high' else "#ffaa00" if sev.lower() == 'medium' else "#76b900"
        st.markdown(f"""<div class="info-card">
<div class="info-title">{risk.get('title', 'N/A')}</div>
<div style="color: #aaaaaa; font-size: 0.9rem; margin-bottom: 10px;">
<strong>Category:</strong> {risk.get('category', 'N/A')} &nbsp;|&nbsp; 
<strong>Severity:</strong> <span style="color: {s_color};">{sev}</span> &nbsp;|&nbsp; 
<strong>Confidence:</strong> {risk.get('confidence_score', 'N/A')}%
</div>
<div class="info-desc">{risk.get('evidence', 'N/A')}</div>
</div>""", unsafe_allow_html=True)
st.divider()

# Section 5: Sentiment Analysis (LLM Zero-Shot)
st.header("Section 5: Sentiment Analysis")
st.markdown("**LLM Zero-Shot Sentiment Classification**")

sentiment_data = data.get('sentiment_analysis', {})
if isinstance(sentiment_data, dict) and sentiment_data:
    try:
        news_score = int(sentiment_data.get('news_sentiment_score', 50))
        pub_score = int(sentiment_data.get('public_sentiment_score', 50))
    except ValueError:
        news_score, pub_score = 50, 50

    c1, c2 = st.columns(2)
    
    def create_gauge(score, title):
        # Determine main bar color based on score
        bar_color = "#ff4b4b" if score < 40 else "#ffaa00" if score <= 60 else "#76b900"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': title, 'font': {'size': 20, 'color': '#ffffff'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#dddddd"},
                'bar': {'color': bar_color},
                'bgcolor': "#1a1a1a",
                'borderwidth': 2,
                'bordercolor': "#333333",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 75, 75, 0.1)'},
                    {'range': [40, 60], 'color': 'rgba(255, 170, 0, 0.1)'},
                    {'range': [60, 100], 'color': 'rgba(118, 185, 0, 0.1)'}
                ],
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="#0e0e0e",
            font={'color': "#ffffff", 'family': "sans-serif"},
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    with c1:
        with st.container(border=True):
            st.plotly_chart(create_gauge(news_score, "Traditional Media"), use_container_width=True)

    with c2:
        with st.container(border=True):
            st.plotly_chart(create_gauge(pub_score, "Public / Developer"), use_container_width=True)
            
    st.info(f"**AI Justification:** {sentiment_data.get('sentiment_justification', 'N/A')}")
else:
    st.warning("Sentiment analysis data not available in the current report.")

st.divider()

# Section 6: Strategic Recommendations
st.header("Section 6: Strategic Recommendations")
recommendations = data.get('recommendations', [])
if isinstance(recommendations, list):
    for idx, rec in enumerate(recommendations):
        if not isinstance(rec, dict): continue
        priority = str(rec.get('priority', 'Medium'))
        p_color = "#ff4b4b" if priority.lower() == 'high' else "#ffaa00" if priority.lower() == 'medium' else "#76b900"

        ev_data = rec.get('supporting_evidence', [])
        if not isinstance(ev_data, list): ev_data = [str(ev_data)]
        ev_list = "".join([f"<li>{e}</li>" for e in ev_data])
        
        imp_data = rec.get('expected_impact', [])
        if not isinstance(imp_data, list): imp_data = [str(imp_data)]
        imp_list = "".join([f"<li>{i}</li>" for i in imp_data])
        
        risk_data = rec.get('risk_assessment', {})
        if not isinstance(risk_data, dict): risk_data = {}
        
        st.markdown(f"""<div class="info-card">
<div style="font-size: 1.4rem; font-weight: bold; color: #ffffff; margin-bottom: 8px;">
Recommendation {idx + 1}: <span style="color: #76b900;">{rec.get('recommendation', 'N/A')}</span>
</div>
<div style="font-size: 1.1rem; margin-bottom: 20px;">
<strong>Priority:</strong> <span style='color:{p_color};'>{priority}</span>
</div>
<div style="display: flex; flex-wrap: wrap; gap: 20px;">
<div style="flex: 1; min-width: 250px;">
<div style="color:#76b900; font-weight:bold; margin-bottom:5px;">Supporting Evidence</div>
<ul style="margin-top:0; padding-left:20px; color:#dddddd;">{ev_list}</ul>
</div>
<div style="flex: 1; min-width: 250px;">
<div style="color:#76b900; font-weight:bold; margin-bottom:5px;">Expected Impact</div>
<ul style="margin-top:0; padding-left:20px; color:#dddddd;">{imp_list}</ul>
</div>
<div style="flex: 1; min-width: 250px;">
<div style="color:#76b900; font-weight:bold; margin-bottom:5px;">Risk Assessment</div>
<ul style="margin-top:0; padding-left:20px; color:#dddddd;">
<li><strong>Financial:</strong> {risk_data.get('financial_risk', 'N/A')}</li>
<li><strong>Operational:</strong> {risk_data.get('operational_risk', 'N/A')}</li>
<li><strong>Strategic:</strong> {risk_data.get('strategic_risk', 'N/A')}</li>
</ul>
</div>
</div>
</div>""", unsafe_allow_html=True)
st.divider()

# Section 7: CEO Briefing
st.header("Section 7: CEO Briefing")
briefing = data.get('ceo_briefing', {})
if not isinstance(briefing, dict): briefing = {}
st.info(f"**What happened?**\n\n{briefing.get('what_happened', 'N/A')}")
st.warning(f"**Why does it matter?**\n\n{briefing.get('why_it_matters', 'N/A')}")
st.success(f"**What should management do next?**\n\n{briefing.get('next_steps', 'N/A')}")
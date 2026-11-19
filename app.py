import json
import streamlit as st
from src.scope_engine import CynbitAIScopeEngine

st.set_page_config(page_title="Cynbit AI Scope & SOW Copilot", page_icon="⚡", layout="wide")

st.title("⚡ Cynbit AI Scope & SOW Copilot (Proof of Concept)")
st.caption("AI/ML Engineer Intern Assessment | Automated Client Brief Analyzer, Architecture Recommender & Effort Estimator")

engine = CynbitAIScopeEngine()

sample_briefs = {
    "Custom Brief (Write Your Own)": "",
    "AI Customer Support + SaaS Portal": "We want to build a multi-tenant B2B SaaS portal with SSO authentication, an AI RAG chatbot trained on our internal help docs, Stripe subscription billing, and an admin analytics dashboard.",
    "FinTech Mobile App with UPI": "Looking for an Android and iOS mobile app for instant UPI payments, user KYC login, real-time push notifications, and merchant settlement reports.",
    "Predictive Churn & BI Pipeline": "We need a data engineering pipeline that syncs HubSpot CRM data, predicts customer churn using ML models, and displays weekly KPIs in an executive dashboard.",
}

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("1. Client Requirement Input")
    preset = st.selectbox("Load Sample Client Inquiry:", list(sample_briefs.keys()))
    default_text = sample_briefs[preset] if preset != "Custom Brief (Write Your Own)" else sample_briefs["AI Customer Support + SaaS Portal"]
    brief = st.text_area("Unstructured Project Brief / RFP Text:", value=default_text, height=170)
    urgency = st.radio("Delivery Priority:", ["Standard", "Accelerated (Fast-Track)"], horizontal=True)
    run_btn = st.button("Generate AI Technical Scope & SOW", type="primary", use_container_width=True)

with col2:
    st.subheader("2. AI-Generated Architecture & Scope")
    if run_btn or brief.strip():
        result = engine.analyze_brief(brief, urgency=urgency)

        m1, m2, m3 = st.columns(3)
        m1.metric("Practice Match", result["service_id"], f"{int(result['retrieval_confidence']*100)}% Confidence")
        m2.metric("Est. Timeline", result["estimated_timeline_weeks"])
        m3.metric("Est. Investment", result["estimated_budget_usd"])

        st.markdown(f"**Matched Cynbit Practice:** `{result['matched_practice']}`")
        st.markdown("**Detected Functional Modules:**")
        st.write(", ".join([f"`{m}`" for m in result["detected_modules"]]))

        st.markdown("**Recommended Tech Stack:**")
        st.write(" • ".join(result["recommended_stack"]))

        st.markdown("**Phased Execution Roadmap:**")
        for step in result["delivery_roadmap"]:
            st.markdown(f"- {step}")

        st.markdown("**Identified Engineering Risks:**")
        for r in result["technical_risks"]:
            st.warning(r)

        with st.expander("View Structured JSON Output (API Payload Ready)"):
            st.json(result)
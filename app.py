import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Telco Retention Engine", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('dashboard_data.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please run 'python model_pipeline.py' first.")
    st.stop()

# Dashboard Title
st.title("📊 Customer Churn & Retention Strategy Engine")
st.markdown("---")

# Executive View
st.header("🏢 Executive View")
col1, col2, col3, col4 = st.columns(4)

high_risk_customers = df[df['Churn_Probability'] > 0.7]
revenue_at_risk = high_risk_customers['Estimated_CLV'].sum()

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Average Churn Risk", f"{df['Churn_Probability'].mean()*100:.1f}%")
col3.metric("High Risk Customers (>70%)", f"{len(high_risk_customers):,}")
col4.metric("Est. Revenue at Risk", f"${revenue_at_risk:,.0f}")

# Visuals
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Risk by Contract Type")
    fig1 = px.box(df, x='Contract', y='Churn_Probability', color='Contract')
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Customer Segmentation")
    fig2 = px.pie(df, names='Segment_Name', title="Customer Distribution by Segment")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Customer View
st.header("👤 Customer Deep-Dive & Retention Recommendations")

# Search mechanism
customer_list = df['customerID'].tolist()
selected_customer = st.selectbox("Search Customer ID:", customer_list)

cust_data = df[df['customerID'] == selected_customer].iloc[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.info(f"**Tenure:** {cust_data['tenure']} months")
    st.info(f"**Contract:** {cust_data['Contract']}")
    st.info(f"**Monthly Charges:** ${cust_data['MonthlyCharges']}")

with c2:
    prob = cust_data['Churn_Probability'] * 100
    if prob > 70:
        st.error(f"**Churn Risk:** {prob:.1f}% (High Risk)")
    elif prob > 40:
        st.warning(f"**Churn Risk:** {prob:.1f}% (Medium Risk)")
    else:
        st.success(f"**Churn Risk:** {prob:.1f}% (Low Risk)")
    
    st.info(f"**Est. CLV:** ${cust_data['Estimated_CLV']:,.2f}")
    st.info(f"**Segment:** {cust_data['Segment_Name']}")

with c3:
    st.subheader("Action to Take:")
    st.success(f"🎯 **{cust_data['Retention_Action']}**")

st.markdown("### Why is this customer at risk? (Key Data)")
st.write(cust_data[['InternetService', 'TechSupport', 'OnlineSecurity', 'DeviceProtection', 'PaymentMethod']].to_frame().T)
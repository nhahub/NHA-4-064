import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# --- Page Config ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🛡️ Fraud Detection Model & Analytics Dashboard")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Analytics", "Fraud Predictor"])

# Path Settings (Adjust if running outside the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
GOLD_DATA_PATH = os.path.join(BASE_DIR, "ingestion", "gold", "master_table")
SILVER_TRANSACTIONS = os.path.join(BASE_DIR, "ingestion", "silver", "transactions_with_labels")

@st.cache_resource
def load_models():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "fraud_detection_model.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_models()

if page == "Analytics":
    st.header("📊 Data Analytics Overview")
    
    st.markdown("**(Note: In a true Big Data environment, we'd query Parquet limits via PySpark or DuckDB. For demo purposes, we're assuming a sampled local Pandas load if data fits.)**")
    
    # Check if Gold data exists to visualize
    if os.path.exists(GOLD_DATA_PATH):
        try:
            # We load the parquet. If it's too big, this might take a moment or we should sample it.
            # Using pandas to read parquet directly for visualization
            df = pd.read_parquet(GOLD_DATA_PATH)
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            total_tx = len(df)
            fraud_tx = df['Target_Num'].sum() if 'Target_Num' in df.columns else 0
            fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0
            
            col1.metric("Total Transactions", f"{total_tx:,}")
            col2.metric("Fraud Cases", f"{int(fraud_tx):,}")
            col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            col4.metric("Avg Transaction", f"${df['amount'].mean():.2f}")
            
            st.divider()
            
            # Visualizations
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Fraud by Hour of Day")
                fraud_by_hour = df.groupby('hour')['Target_Num'].mean().reset_index()
                fig_hour = px.bar(fraud_by_hour, x='hour', y='Target_Num', title="Fraud Rate per Hour", labels={"Target_Num": "Avg Fraud Rate"})
                st.plotly_chart(fig_hour, use_container_width=True)

            with col_b:
                st.subheader("Fraud by Transaction Type")
                # map back the one-hot encoding for visual
                types = []
                for _, row in df.iterrows():
                    if row['is_online'] == 1: types.append("Online")
                    elif row['is_chip'] == 1: types.append("Chip")
                    elif row['is_swipe'] == 1: types.append("Swipe")
                    else: types.append("Unknown")
                df['tx_type'] = types
                
                tx_fraud = df.groupby('tx_type')['Target_Num'].mean().reset_index()
                fig_type = px.bar(tx_fraud, x='tx_type', y='Target_Num', color='tx_type', title="Fraud Rate by Type")
                st.plotly_chart(fig_type, use_container_width=True)

            st.subheader("Correlation Heatmap")
            # select continuous vars
            cols_to_plot = ['amount', 'hour', 'client_mean_amount', 'amount_to_credit_ratio', 'Target_Num']
            corr = df[cols_to_plot].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Selected Features Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading Gold Data: {e}")
    else:
        st.warning("⚠️ Gold Layer data not found! Please run `ingestion/gold.py` first to generate the `master_table` parquet.")

elif page == "Fraud Predictor":
    st.header("🔮 Real-Time Fraud Predictor (XGBoost)")
    
    if model is None or scaler is None:
        st.error("Model files (`fraud_detection_model.pkl` and `scaler.pkl`) not found in the `models/` directory!")
        st.stop()
        
    st.markdown("Enter transaction details below to predict the probability of fraud.")
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0,1,2,3,4,5,6], index=2)
            month = st.slider("Month", 1, 12, 5)
            
        with col2:
            is_night = st.selectbox("Is Night (0-5 AM)?", [0, 1], index=0)
            client_mean_amount = st.number_input("Client Mean Tx Amount", value=50.0)
            amount_to_credit_ratio = st.number_input("Amount to Credit Limit Ratio", value=0.05)
            tx_count_same_day = st.number_input("Tx Count Same Day", min_value=1, value=2)
            
        with col3:
            client_merchant_freq = st.number_input("Client-Merchant Freq", min_value=1, value=5)
            is_online = st.selectbox("Is Online Transaction?", [0, 1], index=1)
            is_chip = st.selectbox("Is Chip Transaction?", [0, 1], index=0)
            has_error = st.selectbox("Experienced Error?", [0, 1], index=0)
            
        submitted = st.form_submit_button("Predict Fraud Probability")
        
        if submitted:
            # Reconstruct feature array exactly as expected by the model
            # Based on DEPI_Project.ipynb:
            # ['amount', 'hour', 'day_of_week', 'month', 'is_night', 'client_mean_amount', 
            #  'amount_to_credit_ratio', 'tx_count_same_day', 'client_merchant_freq', 'is_online', 'is_chip', 'has_error']
            features = pd.DataFrame([{
                'amount': amount,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_night': is_night,
                'client_mean_amount': client_mean_amount,
                'amount_to_credit_ratio': amount_to_credit_ratio,
                'tx_count_same_day': tx_count_same_day,
                'client_merchant_freq': client_merchant_freq,
                'is_online': is_online,
                'is_chip': is_chip,
                'has_error': has_error
            }])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            prob = model.predict_proba(features_scaled)[0][1]
            pred = int(prob >= 0.9885) # The optimal threshold mentioned in the notebook
            
            st.divider()
            if pred == 1:
                st.error(f"🚨 FRAUD DETECTED! Probability: {prob:.4f}")
            else:
                st.success(f"✅ Transaction Approved. Probability of fraud: {prob:.4f}")


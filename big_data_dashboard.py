import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import time

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Big Data Fraud Detection", layout="wide", page_icon="🛡️")

st.title("🌐 Big Data Fraud Detection & Data Lake Dashboard")
st.markdown("لوحة تحكم تفاعلية توضح معمارية البيانات الضخمة (Medallion Architecture) وتحليلات الاحتيال في الوقت الفعلي.")

# إعداد النوافذ الجانبية
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103533.png", width=100)
st.sidebar.header("القائمة الجانبية")
page = st.sidebar.radio("اختر الشاشة", ["معمارية البيانات (Data Lake)", "تحليلات البيانات (Analytics)", "توقع الاحتيال (Fraud Predictor)"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# دالة لحساب حجم المجلدات للـ Data Lake
def get_dir_size(path):
    total_size = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024) # تحويل إلى ميجابايت

@st.cache_resource
def load_models():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_detection_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_models()

if page == "معمارية البيانات (Data Lake)":
    st.header("🗄️ Medallion Architecture Status")
    st.markdown("توضح هذه الشاشة حالة طبقات البيانات الضخمة (Bronze, Silver, Gold).")
    
    col1, col2, col3 = st.columns(3)
    
    bronze_path = os.path.join(BASE_DIR, "ingestion", "bronze")
    silver_path = os.path.join(BASE_DIR, "ingestion", "silver")
    gold_path = os.path.join(BASE_DIR, "ingestion", "gold")
    
    with col1:
        st.info("🥉 Bronze Layer (Raw Data)")
        st.metric("حجم البيانات الكلي", f"{get_dir_size(bronze_path):.2f} MB")
        st.caption("البيانات الخام المستخرجة من قواعد البيانات و APIs.")
        
    with col2:
        st.warning("🥈 Silver Layer (Cleaned Data)")
        st.metric("حجم البيانات الكلي", f"{get_dir_size(silver_path):.2f} MB")
        st.caption("البيانات بعد التنظيف والمعالجة باستخدام PySpark.")
        
    with col3:
        st.success("🥇 Gold Layer (Aggregated Data)")
        st.metric("حجم البيانات الكلي", f"{get_dir_size(gold_path):.2f} MB")
        st.caption("البيانات الجاهزة لتدريب نماذج تعلم الآلة (Master Table).")
        
    st.divider()
    st.subheader("محاكاة تدفق البيانات (Spark Streaming Simulation)")
    if st.button("تشغيل خط البيانات (Run Pipeline)"):
        with st.spinner('يتم معالجة البيانات باستخدام PySpark...'):
            time.sleep(2)
            st.success("تم استيراد البيانات إلى طبقة Bronze بنجاح!")
            time.sleep(2)
            st.success("تم تنظيف البيانات وتحويلها إلى طبقة Silver بنجاح!")
            time.sleep(2)
            st.success("تم تجهيز البيانات للطبقة Gold (Master Table) للتدريب! 🚀")

elif page == "تحليلات البيانات (Analytics)":
    st.header("📊 تحليلات البيانات الضخمة (Big Data Analytics)")
    GOLD_DATA_PATH = os.path.join(BASE_DIR, "ingestion", "gold", "master_table")
    
    if os.path.exists(GOLD_DATA_PATH):
        df = pd.read_parquet(GOLD_DATA_PATH)
        
        col1, col2, col3, col4 = st.columns(4)
        total_tx = len(df)
        fraud_tx = df['Target_Num'].sum() if 'Target_Num' in df.columns else 0
        fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0
        
        col1.metric("إجمالي المعاملات", f"{total_tx:,}")
        col2.metric("حالات الاحتيال", f"{int(fraud_tx):,}")
        col3.metric("معدل الاحتيال", f"{fraud_rate:.2f}%")
        col4.metric("متوسط المعاملة", f"${df['amount'].mean():.2f}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            fraud_by_hour = df.groupby('hour')['Target_Num'].mean().reset_index()
            fig_hour = px.line(fraud_by_hour, x='hour', y='Target_Num', title="توزيع الاحتيال على مدار اليوم", markers=True)
            st.plotly_chart(fig_hour, use_container_width=True)

        with col_b:
            if 'amount' in df.columns:
                fig_amt = px.histogram(df[df['Target_Num']==1], x='amount', nbins=50, title="توزيع مبالغ العمليات الاحتيالية", color_discrete_sequence=['red'])
                st.plotly_chart(fig_amt, use_container_width=True)
                
    else:
        st.warning("⚠️ لا يمكن العثور على طبقة بيانات Gold. يرجى التأكد من تشغيل ملف ingestion/gold.py")

elif page == "توقع الاحتيال (Fraud Predictor)":
    st.header("🔮 نظام التنبؤ الذكي (XGBoost Real-time Inference)")
    
    if model is None or scaler is None:
        st.error("⚠️ فشل في تحميل نموذج تعلم الآلة. يرجى التأكد من وجود ملفات النماذج في مجلد models.")
        st.stop()
        
    st.markdown("قم بإدخال بيانات المعاملة لاختبار الموديل بالوقت الفعلي.")
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("قيمة المعاملة ($)", min_value=0.0, value=150.0)
            hour = st.slider("الساعة", 0, 23, 12)
            day_of_week = st.selectbox("يوم الأسبوع (0=الإثنين, 6=الأحد)", [0,1,2,3,4,5,6], index=2)
            month = st.slider("الشهر", 1, 12, 5)
            
        with col2:
            is_night = st.selectbox("هل المعاملة ليلاً (0-5 AM)؟", [0, 1], index=0)
            client_mean_amount = st.number_input("متوسط إنفاق العميل", value=50.0)
            amount_to_credit_ratio = st.number_input("نسبة المعاملة من الحد الائتماني", value=0.05)
            tx_count_same_day = st.number_input("عدد المعاملات في نفس اليوم", min_value=1, value=2)
            
        with col3:
            client_merchant_freq = st.number_input("تكرار التعامل مع التاجر", min_value=1, value=5)
            is_online = st.selectbox("هل المعاملة أونلاين؟", [0, 1], index=1)
            is_chip = st.selectbox("هل تمت بالشريحة (Chip)؟", [0, 1], index=0)
            has_error = st.selectbox("هل حدث خطأ أثناء العملية؟", [0, 1], index=0)
            
        submitted = st.form_submit_button("توقع احتمالية الاحتيال")
        
        if submitted:
            features = pd.DataFrame([{
                'amount': amount, 'hour': hour, 'day_of_week': day_of_week, 'month': month,
                'is_night': is_night, 'client_mean_amount': client_mean_amount,
                'amount_to_credit_ratio': amount_to_credit_ratio, 'tx_count_same_day': tx_count_same_day,
                'client_merchant_freq': client_merchant_freq, 'is_online': is_online,
                'is_chip': is_chip, 'has_error': has_error
            }])
            
            features_scaled = scaler.transform(features)
            prob = model.predict_proba(features_scaled)[0][1]
            pred = int(prob >= 0.9885)
            
            st.divider()
            if pred == 1:
                st.error(f"🚨 **تنبيه:** احتمالية احتيال عالية جداً! (النسبة: {prob:.4f})")
            else:
                st.success(f"✅ معاملة آمنة. (النسبة: {prob:.4f})")

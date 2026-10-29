import time
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as rt

st.set_page_config(page_title="Smart Factory Edge Dashboard", page_icon="🏭", layout="wide")

st.title("🏭 Smart Factory: Edge Predictive Maintenance Dashboard")
st.markdown("Real-time local inference and sensor telemetry monitoring (Zero cloud dependency).")

# Load ONNX model
@st.cache_resource
def load_model():
    return rt.InferenceSession("machine_fault_model.onnx")

try:
    sess = load_model()
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
except Exception as e:
    st.error(f"Model file not found! Please run `python train.py` first. Error: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("Edge Node Controls")
sim_mode = st.sidebar.selectbox("Simulation Mode", ["Random Live Stream", "Normal Operation", "Inject Fault Precursors"])
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 1.0, 5.0, 2.0)
run_dashboard = st.sidebar.toggle("Start Live Monitoring", value=True)

# Initialize session state for history log
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Vibration", "Pressure", "Temperature", "RPM", "Status"])

placeholder = st.empty()

if run_dashboard:
    # Generate sensor values based on mode
    if sim_mode == "Inject Fault Precursors":
        vib, pres, temp, rpm = np.random.normal(1.7, 0.2), np.random.normal(23.0, 1.0), np.random.normal(85.0, 2.0), np.random.normal(1380, 30)
    elif sim_mode == "Normal Operation":
        vib, pres, temp, rpm = np.random.normal(0.5, 0.08), np.random.normal(30.0, 1.0), np.random.normal(65.0, 2.0), np.random.normal(1500, 15)
    else:
        is_faulty = np.random.choice([0, 1], p=[0.85, 0.15])
        if is_faulty:
            vib, pres, temp, rpm = np.random.normal(1.7, 0.2), np.random.normal(23.0, 1.0), np.random.normal(85.0, 2.0), np.random.normal(1380, 30)
        else:
            vib, pres, temp, rpm = np.random.normal(0.5, 0.08), np.random.normal(30.0, 1.0), np.random.normal(65.0, 2.0), np.random.normal(1500, 15)

    sensor_data = np.array([[vib, pres, temp, rpm]], dtype=np.float32)
    raw_pred = sess.run([output_name], {input_name: sensor_data})[0]
    prediction = int(raw_pred[0])
    status = "FAULT PREDICTED" if prediction == 1 else "NORMAL"

    new_row = {
        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
        "Vibration": round(float(vib), 2),
        "Pressure": round(float(pres), 2),
        "Temperature": round(float(temp), 2),
        "RPM": round(float(rpm), 2),
        "Status": status
    }
    
    # Keep last 30 rows in history
    st.session_state.history = pd.concat([pd.DataFrame([new_row]), st.session_state.history]).head(30)

with placeholder.container():
    if not st.session_state.history.empty:
        latest = st.session_state.history.iloc[0]
        
        # Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Vibration", f"{latest['Vibration']} g")
        col2.metric("Pressure", f"{latest['Pressure']} psi")
        col3.metric("Temperature", f"{latest['Temperature']} °C")
        col4.metric("RPM", f"{latest['RPM']}")
        
        if latest['Status'] == "FAULT PREDICTED":
            col5.error(f"Status: {latest['Status']}")
        else:
            col5.success(f"Status: {latest['Status']}")

        st.markdown("---")

        # Live Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Vibration & Pressure Trend")
            chart_data = st.session_state.history.set_index("Time")[["Vibration", "Pressure"]]
            st.line_chart(chart_data)
        with c2:
            st.subheader("Temperature & RPM Trend")
            chart_data2 = st.session_state.history.set_index("Time")[["Temperature", "RPM"]]
            st.line_chart(chart_data2)

        st.markdown("### Live Edge Telemetry Table")
        st.dataframe(st.session_state.history, use_container_width=True)

        # CSV Download Button in Sidebar
        csv_data = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Export Sensor Logs (CSV)",
            data=csv_data,
            file_name="smart_factory_sensor_logs.csv",
            mime="text/csv"
        )

if run_dashboard:
    time.sleep(refresh_rate)
    st.rerun()
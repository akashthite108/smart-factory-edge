# 🏭 Smart Factory Edge Predictive Maintenance & Monitoring

An industrial edge AI solution designed for smart factories with unreliable network connectivity. This system trains a lightweight machine learning classifier to predict machine breakdowns locally based on real-time sensor telemetry (**vibration, pressure, temperature, and rotational speed**) and deploys it with zero cloud dependency.

---

## 🚀 Key Features
* **Edge Inference:** Runs locally on factory gateways using ONNX Runtime for blazing-fast, deterministic execution.
* **Network Resilience:** Automatically buffers telemetry locally if network connectivity drops and syncs when restored.
* **Interactive Control Room Dashboard:** Real-time Streamlit web dashboard with live charts, metric cards, status alerts, and CSV log export.

---

## 📂 Project Structure
* `train.py` - Generates synthetic industrial data, trains a Random Forest classifier, and exports it to ONNX format.
* `edge_node.py` - Simulates local edge sensor polling, real-time inference, and network fault buffering.
* `dashboard.py` - Streamlit web dashboard for live monitoring and data export.
* `requirements.txt` - Python package dependencies.
* `config.toml` - Custom industrial dark theme configuration for the dashboard.

---

## 🛠️ Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
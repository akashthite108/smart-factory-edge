import time
import logging
import numpy as np
import onnxruntime as rt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("EdgeInferenceModule")

class EdgeFaultDetector:
    def __init__(self, model_path="machine_fault_model.onnx"):
        logger.info("Initializing ONNX Edge Runtime...")
        self.sess = rt.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.local_buffer = []

    def simulate_sensor_stream(self):
        is_faulty = np.random.choice([0, 1], p=[0.85, 0.15])
        if is_faulty:
            vib = np.random.normal(1.7, 0.2)
            pres = np.random.normal(23.0, 1.0)
            temp = np.random.normal(85.0, 2.0)
            rpm = np.random.normal(1380, 30)
        else:
            vib = np.random.normal(0.5, 0.08)
            pres = np.random.normal(30.0, 1.0)
            temp = np.random.normal(65.0, 2.0)
            rpm = np.random.normal(1500, 15)
        return np.array([[vib, pres, temp, rpm]], dtype=np.float32)

    def check_network_status(self):
        return np.random.choice([True, False], p=[0.5, 0.5])

    def run_edge_loop(self):
        logger.info("Edge module running. Monitoring machine telemetry locally...")
        print("Press Ctrl+C to stop the edge module.\n")
        try:
            while True:
                sensor_data = self.simulate_sensor_stream()
                
                # Local Inference
                raw_pred = self.sess.run([self.output_name], {self.input_name: sensor_data})[0]
                prediction = int(raw_pred[0])
                
                payload = {
                    "timestamp": time.time(),
                    "sensors": {
                        "vibration": round(float(sensor_data[0][0]), 2),
                        "pressure": round(float(sensor_data[0][1]), 2),
                        "temperature": round(float(sensor_data[0][2]), 2),
                        "rpm": round(float(sensor_data[0][3]), 2)
                    },
                    "status": "FAIL_PREDICTED" if prediction == 1 else "NORMAL"
                }

                if prediction == 1:
                    logger.warning(f"🚨 LOCAL FAULT FLAGGED! Telemetry: {payload['sensors']}")
                else:
                    logger.info(f"Machine Status: NORMAL | Sensors: {payload['sensors']}")
                
                if self.check_network_status():
                    if self.local_buffer:
                        logger.info(f"Network online. Syncing {len(self.local_buffer)} buffered events to cloud.")
                        self.local_buffer.clear()
                else:
                    logger.warning("⚠️ Network offline. Buffering telemetry locally on edge device.")
                    self.local_buffer.append(payload)

                time.sleep(2.0)
        except KeyboardInterrupt:
            logger.info("Edge module shut down safely.")

if __name__ == "__main__":
    edge_node = EdgeFaultDetector()
    edge_node.run_edge_loop()
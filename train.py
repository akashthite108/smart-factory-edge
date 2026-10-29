import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

def generate_synthetic_factory_data(n_samples=10000, seed=42):
    np.random.seed(seed)
    # Normal operating conditions
    vibration = np.random.normal(0.5, 0.1, int(n_samples * 0.8))
    pressure = np.random.normal(30.0, 1.5, int(n_samples * 0.8))
    temperature = np.random.normal(65.0, 3.0, int(n_samples * 0.8))
    rpm = np.random.normal(1500, 20, int(n_samples * 0.8))
    y_normal = np.zeros(int(n_samples * 0.8))

    # Failure precursor conditions
    n_fail = n_samples - len(y_normal)
    vibration_fail = np.random.normal(1.8, 0.3, n_fail)
    pressure_fail = np.random.normal(22.0, 3.0, n_fail)
    temperature_fail = np.random.normal(88.0, 5.0, n_fail)
    rpm_fail = np.random.normal(1350, 70, n_fail)
    y_fail = np.ones(n_fail)

    X = np.column_stack([
        np.concatenate([vibration, vibration_fail]),
        np.concatenate([pressure, pressure_fail]),
        np.concatenate([temperature, temperature_fail]),
        np.concatenate([rpm, rpm_fail])
    ])
    y = np.concatenate([y_normal, y_fail])
    
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

if __name__ == "__main__":
    print("Generating training data...")
    X, y = generate_synthetic_factory_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    print("\nModel Evaluation Report:")
    print(classification_report(y_test, clf.predict(X_test)))

    print("\nExporting model to ONNX format for edge deployment...")
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = skl2onnx.convert_sklearn(clf, initial_types=initial_type)
    
    with open("machine_fault_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Success! Model saved as 'machine_fault_model.onnx'.")
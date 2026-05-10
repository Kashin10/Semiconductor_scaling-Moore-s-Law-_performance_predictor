import numpy as np
import joblib
import pandas as pd

# ================= LOAD MODELS =================
perf_model = joblib.load("gpu_performance_model.pkl")
bottleneck_model = joblib.load("bottleneck_model.pkl")

# ================= INPUT (SIMULATED FUTURE GPU) =================
gpu = {
    "shader_cores": 20000,
    "memory_bandwidth": 1200,
    "boost_clock": 3000,
    "tdp": 400,
    "transistors": 200,
    "process_nm": 2,
    "die_size": 600
}

# ================= FEATURE ENGINEERING =================
compute_index = gpu["shader_cores"] * gpu["boost_clock"]

memory_pressure_ratio = compute_index / (gpu["memory_bandwidth"] + 1)

perf_per_watt = compute_index / (gpu["tdp"] + 1)

transistor_density = gpu["transistors"] / (gpu["die_size"] + 1)

log_trans = np.log(gpu["transistors"] + 1)

# NOTE: approximate expected trend (simplified)
expected_log = log_trans  # can improve later

moore_dev = log_trans - expected_log

# ================= BUILD FEATURE VECTOR =================
X_perf = pd.DataFrame([{
    "shader_cores": gpu["shader_cores"],
    "memory_bandwidth": gpu["memory_bandwidth"],
    "boost_clock": gpu["boost_clock"],
    "tdp": gpu["tdp"],
    "transistors": gpu["transistors"],
    "process_nm": gpu["process_nm"],
    "die_size": gpu["die_size"],
    "compute_index": compute_index,
    "memory_pressure_ratio": memory_pressure_ratio,
    "perf_per_watt": perf_per_watt,
    "transistor_density": transistor_density,
    "moore_deviation": moore_dev
}])

X_bottle = pd.DataFrame([{
    "shader_cores": gpu["shader_cores"],
    "memory_bandwidth": gpu["memory_bandwidth"],
    "boost_clock": gpu["boost_clock"],
    "tdp": gpu["tdp"],
    "transistors": gpu["transistors"]
}])

# ================= PREDICTIONS =================
pred_perf = perf_model.predict(X_perf)[0]
pred_bottle = bottleneck_model.predict(X_bottle)[0]

# ================= OUTPUT =================
print("\n🚀 FUTURE GPU SIMULATION RESULT")
print("-----------------------------------")
print("Predicted Benchmark Score:", int(pred_perf))
print("Predicted Bottleneck:", pred_bottle)
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ================= LOAD MODELS =================

perf_model = joblib.load("gpu_performance_model.pkl")
bottleneck_model = joblib.load("bottleneck_model.pkl")
fps_model = joblib.load("fps_model.pkl")

# ================= LOAD DATA =================

fps_data = pd.read_csv("fps_flat.csv")
gpu_data = pd.read_csv("gpu_ml_dataset.csv")

games = sorted(fps_data["game"].unique())
resolutions = sorted(fps_data["resolution"].unique())

# ================= BRAND DETECTION =================

def get_brand(name):
    name = str(name).lower()
    if "rtx" in name or "gtx" in name:
        return "NVIDIA"
    elif "rx" in name or "radeon" in name:
        return "AMD"
    elif "arc" in name or "intel" in name:
        return "Intel"
    else:
        return "Other"

gpu_data["brand"] = gpu_data["gpu_name"].apply(get_brand)

# ================= UI =================

st.title("🚀 GPU Performance & FPS Simulator")

# ================= BRAND FILTER =================

brand = st.selectbox("Select Brand", ["All", "NVIDIA", "AMD", "Intel"])

if brand == "All":
    filtered_gpus = gpu_data
else:
    filtered_gpus = gpu_data[gpu_data["brand"] == brand]

gpu_names = sorted(filtered_gpus["gpu_name"].unique())

# ================= GPU SELECT =================

selected_gpu = st.selectbox("Choose GPU (optional)", ["Custom"] + gpu_names)

# ================= DEFAULT VALUES =================

if selected_gpu != "Custom":
    gpu_row = filtered_gpus[filtered_gpus["gpu_name"] == selected_gpu].iloc[0]

    default_shader = int(gpu_row["shader_cores"])
    default_bw = int(gpu_row["memory_bandwidth"])
    default_clock = int(gpu_row["boost_clock"])
    default_tdp = int(gpu_row["tdp"])
    default_trans = int(gpu_row["transistors"])
    default_process = int(gpu_row["process_nm"])
    default_die = int(gpu_row["die_size"])

    st.success(f"Loaded specs for: {selected_gpu}")

else:
    default_shader = 5000
    default_bw = 500
    default_clock = 2000
    default_tdp = 250
    default_trans = 50
    default_process = 7
    default_die = 400

# ================= GPU INPUT (SLIDERS) =================

shader_cores = st.slider("Shader Cores", 100, 30000, default_shader)
memory_bandwidth = st.slider("Memory Bandwidth (GB/s)", 10, 2000, default_bw)
boost_clock = st.slider("Boost Clock (MHz)", 500, 4000, default_clock)
tdp = st.slider("TDP (Watts)", 10, 1000, default_tdp)
transistors = st.slider("Transistors (Billions)", 1, 300, default_trans)
process_options = [14, 10, 8, 7, 6, 5, 4, 3, 2]

if default_process in process_options:
    index_val = process_options.index(default_process)
else:
    # fallback to closest value
    closest = min(process_options, key=lambda x: abs(x - default_process))
    index_val = process_options.index(closest)

process_nm = st.selectbox(
    "Process Node (nm)",
    process_options,
    index=index_val
)
die_size = st.slider("Die Size (mm²)", 50, 1000, default_die)

st.caption("💡 You can modify values even after selecting a GPU")

# ================= GAME INPUT =================

game = st.selectbox("Select Game", games)
resolution = st.selectbox("Resolution", resolutions)

# ================= FEATURE ENGINEERING =================

compute_index = shader_cores * boost_clock

# ✅ normalized version
memory_pressure_ratio = (compute_index / (memory_bandwidth + 1)) / 1000

perf_per_watt = compute_index / (tdp + 1)
transistor_density = transistors / (die_size + 1)

log_trans = np.log(transistors + 1)
expected_log = log_trans
moore_dev = log_trans - expected_log

# ================= BUTTON =================

if st.button("Run Simulation"):

    # ---------- Performance ----------
    X_perf = [[
        shader_cores,
        memory_bandwidth,
        boost_clock,
        tdp,
        transistors,
        process_nm,
        die_size,
        compute_index,
        memory_pressure_ratio,
        perf_per_watt,
        transistor_density,
        moore_dev
    ]]

    benchmark = perf_model.predict(X_perf)[0]

    # ---------- Bottleneck ----------
    X_bottle = [[
        shader_cores,
        memory_bandwidth,
        boost_clock,
        tdp,
        transistors
    ]]

    bottleneck = bottleneck_model.predict(X_bottle)[0]

    # ---------- FPS ----------
    X_fps = pd.DataFrame([{
        "shader_cores": shader_cores,
        "memory_bandwidth": memory_bandwidth,
        "boost_clock": boost_clock,
        "tdp": tdp,
        "transistors": transistors,
        "compute_index": compute_index,
        "memory_pressure_ratio": memory_pressure_ratio,
        "perf_per_watt": perf_per_watt,
        "transistor_density": transistor_density,
        "moore_deviation": moore_dev,
        "game": game,
        "resolution": resolution
    }])

    fps = fps_model.predict(X_fps)[0]

    # ================= OUTPUT =================

    st.subheader("📊 Results")

    st.metric("Benchmark Score", int(benchmark))
    st.metric("Predicted FPS", round(fps, 1))

    if bottleneck == "memory_bound":
        st.warning("⚠️ Memory Bottleneck Detected")
        st.write("Memory Pressure: High ⚠️")
    else:
        st.success("✅ Compute Bound")
        st.write("Memory Pressure: Low / Balanced ✅")

    st.markdown("---")

    st.write("### 🔍 System Insights")

    st.write(f"Compute Index: {int(compute_index)}")

    # FPS Quality
    if fps > 120:
        st.success("🔥 Ultra Smooth Performance")
    elif fps > 60:
        st.info("🎮 Smooth Gameplay")
    elif fps > 40:
        st.warning("⚠️ Playable but not smooth")
    else:
        st.error("❌ Poor Performance")

    st.write(f"Perf/Watt: {round(perf_per_watt, 2)}")

    st.markdown("### 🤖 Why this result?")

    if bottleneck == "memory_bound":
        st.write("Performance limited by memory bandwidth at this resolution.")
    else:
        st.write("Performance limited by compute capability (shader throughput).")
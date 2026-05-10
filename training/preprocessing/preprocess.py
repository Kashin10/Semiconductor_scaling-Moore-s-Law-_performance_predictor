import pandas as pd
import numpy as np
import re

# ================= LOAD =================
gpu = pd.read_csv("gpu_1986-2026.csv")

# ================= BASIC CLEAN =================

gpu = gpu.rename(columns={
    "Name": "gpu_name",
    "Graphics Card__Announced": "year"
})

def clean(x):
    x = str(x).lower()
    x = re.sub(r'nvidia|amd|geforce|radeon', '', x)
    x = re.sub(r'[^a-z0-9 ]', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

gpu["gpu_name"] = gpu["gpu_name"].apply(clean)

# ================= YEAR =================

def extract_year(x):
    x = str(x)
    m = re.search(r'(19|20)\d{2}', x)
    return int(m.group()) if m else np.nan

gpu["year"] = gpu["year"].apply(extract_year)

# ================= FEATURE SELECTION =================

features = [
    "gpu_name",
    "year",
    "Render Config__Shading Units",
    "Memory__Bandwidth",
    "Clock Speeds__Boost Clock",
    "Board Design__TDP",
    "Graphics Processor__Transistors",
    "Graphics Processor__Process Size",
    "Graphics Processor__Die Size"
]

gpu = gpu[features]

gpu.columns = [
    "gpu_name",
    "year",
    "shader_cores",
    "memory_bandwidth",
    "boost_clock",
    "tdp",
    "transistors",
    "process_nm",
    "die_size"
]

# ================= NUMERIC EXTRACTION =================

def num(x):
    x = str(x)
    m = re.search(r'\d+\.?\d*', x)
    return float(m.group()) if m else np.nan

for c in gpu.columns:
    if c != "gpu_name":
        gpu[c] = gpu[c].apply(num)

# ================= IMPUTATION =================

# year → interpolate trend
gpu["year"] = gpu["year"].interpolate()

# architecture features → median
for c in [
    "shader_cores",
    "memory_bandwidth",
    "boost_clock",
    "tdp",
    "transistors",
    "process_nm",
    "die_size"
]:
    gpu[c] = gpu[c].fillna(gpu[c].median())

print("Final Shape:", gpu.shape)
print(gpu.head())

gpu.to_csv("clean_gpu_specs.csv", index=False)
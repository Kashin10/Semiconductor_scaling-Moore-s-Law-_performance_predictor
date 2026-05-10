import pandas as pd
import numpy as np

df = pd.read_csv("clean_gpu_specs.csv")
print(df.head())

print("Initial Shape:", df.shape)

# Proxy for compute throughput
df["compute_index"] = df["shader_cores"] * df["boost_clock"]

# ================= MEMORY PRESSURE =================
# High = memory bottleneck
df["memory_pressure_ratio"] = (
    df["compute_index"] / (df["memory_bandwidth"] + 1) / 1000
)

# ================= PERFORMANCE PER WATT =================
df["perf_per_watt"] = (
    df["compute_index"] / (df["tdp"] + 1)
)

# ================= TRANSISTOR DENSITY =================
df["transistor_density"] = (
    df["transistors"] / (df["die_size"] + 1)
)






# ================= MOORE'S LAW MODEL =================

# log transform
df["log_transistors"] = np.log(df["transistors"] + 1)

# USE CLEAN SUBSET FOR FITTING ONLY
fit_df = df.dropna(subset=["year", "log_transistors"])

# fit linear model safely
coef = np.polyfit(fit_df["year"], fit_df["log_transistors"], 1)

# apply to full dataset
df["expected_log_transistors"] = coef[0] * df["year"] + coef[1]

# deviation
df["moore_deviation"] = (
    df["log_transistors"] - df["expected_log_transistors"]
)



# ================= OPTIONAL NORMALIZATION =================
# Avoid extreme scale differences
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(df[col].median())



# ================= FINAL CHECK =================
print("\nFinal Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nSample Data:")
print(df.head())

print("\nStatistics:")
print(df.describe())


df.to_csv("gpu_ml_dataset.csv", index=False)
print("\nFeature Engineering Complete. Saved as gpu_ml_dataset.csv")
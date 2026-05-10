import pandas as pd
from difflib import get_close_matches


df = pd.read_csv("gpu_ml_dataset.csv")
bench = pd.read_csv("GPU_benchmarks_v7.csv")

# clean names (same logic)
def clean(x):
    import re
    x = str(x).lower()
    x = re.sub(r'nvidia|amd|geforce|radeon', '', x)
    x = re.sub(r'[^a-z0-9 ]', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

df["gpu_name"] = df["gpu_name"].apply(clean)
bench["gpuName"] = bench["gpuName"].apply(clean)

bench_names = bench["gpuName"].tolist()


#fuzzy matching gpu names
def match_gpu(name):
    match = get_close_matches(name, bench_names, n=1, cutoff=0.6)
    return match[0] if match else None

print("Matching GPUs...")

df["matched_name"] = df["gpu_name"].apply(match_gpu)

#merge both datasets
merged = df.merge(
    bench,
    left_on="matched_name",
    right_on="gpuName",
    how="left"
)

#rename benchmark column
merged = merged.rename(columns={
    "G3Dmark": "benchmark_score"
})

# keep only rows where benchmark exists
merged = merged.dropna(subset=["benchmark_score"])

print("Final Shape after merge:", merged.shape)
print(merged[["gpu_name","benchmark_score"]].head())

merged.to_csv("gpu_with_benchmark.csv", index=False)

print("✅ Benchmark merge complete")
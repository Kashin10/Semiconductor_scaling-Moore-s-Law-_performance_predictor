import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import joblib

# ================= LOAD =================

fps = pd.read_csv("fps_flat.csv")
gpu = pd.read_csv("gpu_ml_dataset.csv")

# ================= CLEAN NAMES =================

def clean(x):
    x = str(x).lower()
    x = re.sub(r'nvidia|amd|geforce|radeon', '', x)
    x = re.sub(r'[^a-z0-9 ]', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

fps["gpu_name"] = fps["gpu_name"].apply(clean)
gpu["gpu_name"] = gpu["gpu_name"].apply(clean)

# ================= MERGE =================

df = fps.merge(gpu, on="gpu_name", how="inner")

print("Merged Shape:", df.shape)

# ================= FEATURES =================

num_features = [
    "shader_cores",
    "memory_bandwidth",
    "boost_clock",
    "tdp",
    "transistors",
    "compute_index",
    "memory_pressure_ratio",
    "perf_per_watt",
    "transistor_density",
    "moore_deviation"
]

cat_features = ["game", "resolution"]

X = df[num_features + cat_features]
y = df["fps"]

# ================= PREPROCESS =================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ],
    remainder="passthrough"
)

# ================= MODEL =================

model = Pipeline([
    ("prep", preprocessor),
    ("reg", XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])

# ================= TRAIN =================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training FPS model...")
model.fit(X_train, y_train)

# ================= EVALUATE =================

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nFPS Model Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# ================= SAVE =================

joblib.dump(model, "fps_model.pkl")

print("\n✅ FPS model saved.")
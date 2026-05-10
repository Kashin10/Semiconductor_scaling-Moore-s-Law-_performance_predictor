import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np


df = pd.read_csv("gpu_with_benchmark.csv")

print("Dataset Shape:", df.shape)


features = [
    "shader_cores",
    "memory_bandwidth",
    "boost_clock",
    "tdp",
    "transistors",
    "process_nm",
    "die_size",
    "compute_index",
    "memory_pressure_ratio",
    "perf_per_watt",
    "transistor_density",
    "moore_deviation"
]

X = df[features]
y = df["benchmark_score"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = XGBRegressor(  #XGBoost combines multiple decision trees
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)



importances = model.feature_importances_  #feature_importances_ is a built in attribute  of a TRAINED model

feat_imp = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feat_imp)


import joblib
joblib.dump(model, "gpu_performance_model.pkl")

print("\n✅ Model training complete and saved.")
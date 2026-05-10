import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("gpu_ml_dataset.csv")
print("Dataset shape:" , df.shape)

# ================= CREATE LABEL =================
# threshold can be tuned
threshold = df["memory_pressure_ratio"].median()

def classify(row):
    if row["memory_pressure_ratio"] > threshold:
        return "memory_bound"
    else:
        return "compute_bound"

df["bottleneck"] = df.apply(classify, axis=1)
print("\nBottleneck Distribution:")
print(df["bottleneck"].value_counts())



# ================= FEATURES =================
features = [
    "shader_cores",
    "memory_bandwidth",
    "boost_clock",
    "tdp",
    "transistors"
]

X = df[features]
y = df["bottleneck"]

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ================= MODEL =================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
print("\nTraining bottleneck model...")
model.fit(X_train, y_train)



# ================= EVALUATION =================
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feat_imp = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feat_imp)



# ================= SAVE MODEL =================
import joblib
joblib.dump(model, "bottleneck_model.pkl")

print("\n✅ Bottleneck model saved.")

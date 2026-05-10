
from fastapi import FastAPI

app = FastAPI(
    title="ScaleSage-GPP",
    version="1.0"
)

@app.get("/")

def home():

    return {
        "message":
        "ScaleSage-GPP Backend Running"
    }

@app.post("/predict")

def predict():

    return {
        "fps_prediction": 144,
        "confidence": 0.98,
        "top_feature":
        "Transistor Density"
    }

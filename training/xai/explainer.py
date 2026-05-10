
def explain_prediction():

    return {

        "SHAP":
        [
            ("Transistor Density", 0.42),
            ("Process Node", 0.31)
        ],

        "LIME":
        [
            ("VRAM", 0.14),
            ("Memory Bandwidth", 0.11)
        ],

        "S-IG":
        [
            ("Scaling Trajectory", 0.56)
        ]
    }

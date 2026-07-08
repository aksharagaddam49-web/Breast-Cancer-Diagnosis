from tensorflow.keras.models import load_model

for model_name in ["MobileNet.h5", "Xception.h5", "cancer.h5"]:
    try:
        print(f"Loading {model_name}")
        load_model(model_name, compile=False)
        print(f"{model_name} OK")
    except Exception as e:
        print(f"{model_name} FAILED")
        print(e)
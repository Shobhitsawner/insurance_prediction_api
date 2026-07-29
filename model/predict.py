import pickle
import numpy as np
import pandas as pd
from schema.user_input import UserInput

# Load ML Model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_output(data: UserInput):
    input_data = {
        "age": data.age,
        "weight": data.weight,
        "height": data.height,
        "smoker": data.smoker,
        "city": data.city,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
        "bmi": data.bmi,
        "lifestyle_risk": data.lifestyle_risk,
        "age_group": data.age_group,
        "city_tier": data.city_tier,
    }

    input_df = pd.DataFrame([input_data])

    # 1. Get predicted category
    prediction = model.predict(input_df)[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    # 2. Get probabilities & calculate confidence score
    confidence_score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        max_prob = float(np.max(probabilities))
        # Formatted to 2 decimal places percentage (e.g., 0.875 -> 87.5)
        confidence_score = round(max_prob * 100, 2)

    return prediction, confidence_score
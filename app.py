from fastapi import FastAPI
from fastapi.responses import JSONResponse
from model.predict import predict_output
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
app = FastAPI(title="Insurance Premium Prediction API")


@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):
    try:
        prediction, confidence_score = predict_output(data)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "predicted_category": prediction,
                "confidence_score": (
                    f"{confidence_score}%"
                    if confidence_score
                    else "N/A"
                ),
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal Server Error: {str(e)}",
            },
        )
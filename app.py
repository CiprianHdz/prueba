from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("trained_model.pkl")

# Define the app
app = FastAPI()

# Define a request body schema
# to do: some variables are in fact int - update
class PredictionRequest(BaseModel):
    bar: float
    baz: float
    xgt: float
    qgg: float
    lux: float
    wsg: float
    yyz: float
    drt: float
    gox: float
    foo: float
    boz: float
    fyt: float
    lgh: float
    hrt: float
    juu: float

# Endpoint for prediction
@app.post("/predict/")
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.dict()])
    prediction =  model.predict(input_df)
    return {"prediction": float(prediction)}

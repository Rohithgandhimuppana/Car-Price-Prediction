from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import os

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load your pipeline ONCE (good performance)
pipeline = joblib.load("car_predict_pipeline.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Brand: str = Form(...),
    model_name: str = Form(...),
    Year: int = Form(...),
    Kilometers_Driven: float = Form(...),
    Fuel_Type: str = Form(...),
    Engine: float = Form(...)
):
    # Create input dataframe
    input_df = pd.DataFrame({
        "Year": [Year],
        "Kilometers_Driven": [Kilometers_Driven],
        "Engine": [Engine],
        "Brand": [Brand],
        "Model": [model_name],
        "Fuel_Type": [Fuel_Type]
    })

    # Predict
    prediction = pipeline.predict(input_df)[0]

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": round(prediction, 2)}
    )

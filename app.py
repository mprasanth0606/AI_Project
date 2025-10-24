import pickle
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# ---- Load trained model ----
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# ---- Set up templates ----
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predicted_price": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, area: float = Form(...), bedrooms: int = Form(...), age: int = Form(...)):
    new_data = pd.DataFrame([[area, bedrooms, age]], columns=['area', 'bedrooms', 'age'])
    predicted_price = model.predict(new_data)[0]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_price": f"{predicted_price:,.2f}",
        "area": area,
        "bedrooms": bedrooms,
        "age": age
    })

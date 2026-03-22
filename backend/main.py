from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

# Frontend (React) namma API ah access panna CORS allow pannanum
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Model irukka edathoda path ah set pandrom
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ai_model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ai_model", "vectorizer.pkl")

print("🔄 Loading AI Model into API...")
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}. Please check if .pkl files exist.")

# User anuppura Data-voda format
class NewsInput(BaseModel):
    text: str

# API work aagudha nu check panna oru basic route
@app.get("/")
def read_root():
    # Idhu direct-ah namma HTML file-ah browser-kku anuppidum
    return FileResponse("index.html")

# Main Prediction Route
@app.post("/predict")
def predict_news(news: NewsInput):
    # 1. User kudutha text-ah numbers ah maathurom
    vectorized_text = vectorizer.transform([news.text])
    
    # 2. Model kitta kuduthu unmaiya poiya nu kandupudikirom
    prediction = model.predict(vectorized_text)[0]
    
    # 3. Result-ah thiruppi anupurom
    return {
        "prediction": prediction,
        "status": "success"
    }
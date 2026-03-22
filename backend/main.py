from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATABASE SETUP ----------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./news_history.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Table Eppadi Irukkanum nu solrom
class HistoryDB(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    news_text = Column(String, index=True)
    prediction = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Table-ah create pandrom
Base.metadata.create_all(bind=engine)

# Database-ah open panni close pandra function
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ai_model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ai_model", "vectorizer.pkl")

print("🔄 Loading AI Model...")
model = pickle.load(open(MODEL_PATH, 'rb'))
vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
print("✅ Model Loaded!")

class NewsInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return FileResponse("index.html")

# PREDICT API (With Database Save)
@app.post("/predict")
def predict_news(news: NewsInput, db: Session = Depends(get_db)):
    vectorized_text = vectorizer.transform([news.text])
    prediction = model.predict(vectorized_text)[0]
    
    # Database-la save pandrom!
    db_record = HistoryDB(news_text=news.text, prediction=prediction)
    db.add(db_record)
    db.commit()
    
    return {"prediction": prediction, "status": "success"}

# PUDHU API: History edukkuka
@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    # Kadaisiya theduna 5 news-ah eduthu anupurom
    records = db.query(HistoryDB).order_by(HistoryDB.id.desc()).limit(5).all()
    return records
# 🕵️‍♂️ Fake News Detector - AI Powered Enterprise Solution

An end-to-end Full-Stack Web Application that uses **Machine Learning** to detect whether a news article is **REAL** or **FAKE**. This project is built using **FastAPI**, **Scikit-Learn**, and **SQLite**.

## 🚀 Features
- **AI Prediction:** Uses a Passive Aggressive Classifier trained on 72,000+ news articles.
- **Enterprise UI:** Clean, responsive interface with professional disclaimers.
- **Search History:** Automatically saves your recent searches to a local **SQLite Database**.
- **Transparency:** Clearly mentions model technology and 80-90% accuracy range.

## 🛠️ Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
- **Backend:** Python, FastAPI, Uvicorn
- **AI/ML:** Scikit-Learn, Pandas, NumPy, TF-IDF Vectorization
- **Database:** SQLite (SQLAlchemy ORM)

## 📊 Dataset
The model is trained on the **WELFake_Dataset** from Kaggle, which contains a balanced collection of 72,134 news articles (35,028 Real and 37,106 Fake).

## 📂 Project Structure
```text
FakeNews_Detector/
├── ai_model/
│   ├── train.py              # ML Training Script
│   ├── WELFake_Dataset.csv   # Dataset (Ignored in Git)
│   ├── fake_news_model.pkl   # Saved AI Model
│   └── vectorizer.pkl        # Saved TF-IDF Vectorizer
├── backend/
│   ├── main.py               # FastAPI Server Logic
│   ├── index.html            # Frontend UI
│   └── news_history.db       # SQLite Database file
└── .gitignore                # Files to ignore (large models/datasets)

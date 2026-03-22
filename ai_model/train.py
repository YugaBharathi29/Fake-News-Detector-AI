import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

print("🚀 Step 1: Loading WELFake Dataset (Idhu konjam time edukkum, wait pannunga)...")
# Unga puthu dataset file name
df = pd.read_csv('WELFake_Dataset.csv') 

print("🧹 Step 2: Cleaning Data...")
# Empty values irundha adha remove pannidrom
df = df.dropna()

# WELFake la label 0 (Fake) and 1 (Real) nu irukkum. 
# Adha namma UI-kku yetha madhiri text-ah maathurom.
df['label'] = df['label'].map({0: 'FAKE', 1: 'REAL'})

print("✂️ Step 3: Splitting 72,000+ articles for Training and Testing...")
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print("🔢 Step 4: Converting Text to Numbers (TF-IDF)...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

print("🧠 Step 5: Training the Enterprise AI Model...")
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

print("📊 Step 6: Checking Model Accuracy...")
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'✅ Enterprise Model Accuracy: {round(score*100,2)}%')

print("💾 Step 7: Saving the Big Model...")
pickle.dump(pac, open('fake_news_model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))

print("🎉 Success! Your Real AI Model is ready!")
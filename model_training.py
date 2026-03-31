import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("Loading dataset...")

df = pd.read_excel("data/dataset.csv.xlsx")

print("Cleaning data...")

# Remove rows where Comment or Label is empty
df = df[['Comment', 'Label']].dropna()

X = df['Comment']
y = df['Label']

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Vectorizing text...")

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Evaluating model...\n")

y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred))
import joblib

print("\nSaving model...")

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")
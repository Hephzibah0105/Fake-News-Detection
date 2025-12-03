import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_csv("dataset.csv")
x = data['text']
y = data['label']

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

# MODEL
model = PassiveAggressiveClassifier()
model.fit(x_train, y_train)

# Prediction
pred = model.predict(x_test)
score = accuracy_score(y_test, pred)

print("Accuracy:", score)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

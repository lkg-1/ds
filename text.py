import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


df = pd.read_csv("textpro.csv")


x = df["sanitized_property_summary"]
y = df["type"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


tfidf = TfidfVectorizer()
x_train_tfidf = tfidf.fit_transform(x_train)


model = MultinomialNB()
model.fit(x_train_tfidf, y_train)


x_test_tfidf = tfidf.transform(x_test)
y_pred = model.predict(x_test_tfidf)


print("Accuracy:", accuracy_score(y_test, y_pred))


user = ["4 bedrooms with hall"]
user_tfidf = tfidf.transform(user)
result = model.predict(user_tfidf)

print("Prediction:", result)

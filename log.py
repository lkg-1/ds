import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("reg.csv")

print(df.head())
print(df.describe())
print(df.info())

# ------------------ FEATURES ------------------
df_numeric = df.select_dtypes(include=['number'])

# Fill missing values
df_numeric = df_numeric.fillna(df_numeric.mean())

X = df_numeric
#X = df_numeric.drop(columns=["addicted_label"])

# ------------------ TARGET ENCODING ------------------
le = LabelEncoder()
y = le.fit_transform(df["academic_work_impact"])

# ------------------ SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ SCALING (ADDED) ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train
X_test = scaler.transform(X_test)         # transform test

# ------------------ MODEL ------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ------------------ PREDICTION ------------------
y_pred = model.predict(X_test)

# ------------------ METRICS ------------------
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall   :", recall_score(y_test, y_pred, average='macro'))
print("F1 Score :", f1_score(y_test, y_pred, average='macro'))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_))


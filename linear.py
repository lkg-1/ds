import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load your dataset
df = pd.read_csv("your_file.csv")

print(df.head())
print(df.info())


# OPTIONAL: handle missing values
df = df.fillna(df.mean(numeric_only=True))


# If any categorical columns exist → encode
df = pd.get_dummies(df, drop_first=True)


# Set target (must be numeric)
target_column = "your_target"   # <-- CHANGE THIS

X = df.drop(target_column, axis=1)
y = df[target_column]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model
model = LinearRegression()
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2  :", r2_score(y_test, y_pred))


# Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("telecom_churn.csv")

# Show first rows
print(df.head())

# Check missing values
print("\nMissing Values")
print(df.isnull().sum())

# Fill missing numeric values with mean
df = df.fillna(df.mean(numeric_only=True))

# Remove duplicate rows
df = df.drop_duplicates()

# Convert categorical columns to numeric (encoding)
df = pd.get_dummies(df, drop_first=True)

# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

# -------- OUTLIER HANDLING (IQR METHOD) --------
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.clip(df[col], lower, upper)

# -------- STANDARDIZATION --------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save processed dataset
df.to_csv("processed_dataset.csv", index=False)

print("\nPreprocessing + Outlier Handling + Standardization Completed")

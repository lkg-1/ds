import pandas as pd
# Load dataset
df = pd.read_csv("telecom_churn.csv")
# View first rows
print("Dataset Preview")
print(df.head())
# Statistical summary
print("\nSummary Statistics")
print(df.describe())

# Measures of Central Tendency
print("\nMean")
print(df.mean(numeric_only=True))

print("\nMedian")
print(df.median(numeric_only=True))

print("\nMode")
print(df.mode().iloc[0])

# Measures of Dispersion
print("\nVariance")
print(df.var(numeric_only=True))

print("\nStandard Deviation")
print(df.std(numeric_only=True))

print("\nRange")
print(df.max(numeric_only=True) - df.min(numeric_only=True))

# Shape of Distribution
print("\nSkewness")
print(df.skew(numeric_only=True))

print("\nKurtosis")
print(df.kurt(numeric_only=True))


import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display column names
print("COLUMN NAMES:")
print(df.columns.tolist())

# Count employees who stayed and left
print("\nATTRITION COUNT:")
print(df["Attrition"].value_counts())

# Calculate percentages
print("\nATTRITION PERCENTAGE:")
print(df["Attrition"].value_counts(normalize=True) * 100)

# Display basic information
print("\nDATASET INFORMATION:")
print(df.info())

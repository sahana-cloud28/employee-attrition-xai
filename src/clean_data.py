
import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Remove unnecessary columns
df = df.drop(columns=[
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
])

# Check remaining columns
print("Cleaned dataset shape:", df.shape)

print("\nRemaining columns:")
print(df.columns.tolist())

# Save cleaned dataset
df.to_csv("data/cleaned_employee_attrition.csv", index=False)

print("\nCleaned dataset saved successfully!")

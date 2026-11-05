import pandas as pd

df = pd.read_csv("data/cleaned_employee_attrition.csv")

print("CATEGORICAL FEATURES:")

for col in df.select_dtypes(include="object").columns:
    if col != "Attrition":
        print(f"\n{col}:")
        print(df[col].unique())

print("\nNUMERICAL FEATURES:")

for col in df.select_dtypes(include="number").columns:
    print(f"{col}: Minimum = {df[col].min()}, Maximum = {df[col].max()}")
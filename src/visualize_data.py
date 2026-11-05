
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_employee_attrition.csv")

sns.set_style("whitegrid")

# 1. Attrition Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.savefig("data/attrition_distribution.png")
plt.show()

# 2. Attrition by Overtime
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="OverTime", hue="Attrition")
plt.title("Attrition by Overtime")
plt.savefig("data/attrition_overtime.png")
plt.show()

# 3. Attrition by Age
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Age", hue="Attrition", multiple="stack")
plt.title("Attrition by Age")
plt.savefig("data/attrition_age.png")
plt.show()

# 4. Attrition by Job Satisfaction
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="JobSatisfaction", hue="Attrition")
plt.title("Attrition by Job Satisfaction")
plt.savefig("data/attrition_satisfaction.png")
plt.show()

print("All graphs created successfully!")


import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/cleaned_employee_attrition.csv")

# Separate input and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Decision Tree
model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=5
)

model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Explain test data
shap_values = explainer.shap_values(X_test)

# Display feature importance
shap.summary_plot(shap_values, X_test)

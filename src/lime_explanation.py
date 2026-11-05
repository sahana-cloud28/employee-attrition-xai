
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from lime.lime_tabular import LimeTabularExplainer

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

# Create LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Stayed", "Left"],
    mode="classification"
)

# Explain one employee
employee = X_test.iloc[0].values

explanation = explainer.explain_instance(
    employee,
    model.predict_proba,
    num_features=10
)

# Display explanation
print("Actual Attrition:", y_test.iloc[0])
print("Predicted Attrition:", model.predict([employee])[0])

explanation.save_to_file("data/lime_explanation.html")

print("LIME explanation saved successfully!")
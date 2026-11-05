import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/cleaned_employee_attrition.csv")

# Separate input and output
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Convert text columns into numbers
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=5
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("DECISION TREE RESULTS")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
joblib.dump(model, "models/decision_tree_model.pkl")

print("Model saved successfully!")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("Feature names saved successfully!")
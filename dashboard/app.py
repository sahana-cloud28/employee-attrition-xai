from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

[data-testid="stMetricValue"] {
    color: #60a5fa;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p {
    color: #f8fafc;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #cbd5e1;
}

[data-testid="stSidebar"] label {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# Custom dark dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #F8FAFC;
    }

    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 700;
    }

    p, label {
        color: #CBD5E1;
    }

    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }

    [data-testid="stForm"] {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Load model and feature names
decision_tree_model = joblib.load("models/decision_tree_model.pkl")

random_forest_model = joblib.load("models/random_forest_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Load dataset to get valid input ranges and categories
df = pd.read_csv("data/cleaned_employee_attrition.csv")

# Prepare training data for LIME
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"Yes": 1, "No": 0})

X = pd.get_dummies(X, drop_first=True, dtype=int)

# Match the features used by the trained model
X = X.reindex(columns=feature_names, fill_value=0)

# Recreate the training split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create LIME explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["Stayed", "Left"],
    mode="classification"
)

employee_data = df.drop(columns=["Attrition"])

categorical_columns = employee_data.select_dtypes(
    include=["str", "object"]
).columns.tolist()

numerical_columns = [
    col for col in employee_data.columns
    if col not in categorical_columns
]
# ==========================================
# PROFESSIONAL SIDEBAR DESIGN
# ==========================================

st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0F172A 0%,
        #111C31 100%
    );
    border-right: 1px solid #263449;
}

/* Sidebar padding */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* Navigation heading */
[data-testid="stSidebar"] h2 {
    color: #FFFFFF;
}

/* Radio navigation spacing */
[data-testid="stSidebar"] .stRadio > div {
    gap: 9px;
}

/* Navigation items */
[data-testid="stSidebar"] .stRadio label {
    background: #19263B;
    border: 1px solid #263750;
    border-radius: 10px;
    padding: 11px 14px;
    transition: all 0.2s ease;
}

/* Hover effect */
[data-testid="stSidebar"] .stRadio label:hover {
    background: #243B5A;
    border-color: #3B82F6;
}

/* Radio text */
[data-testid="stSidebar"] .stRadio label p {
    color: #E2E8F0 !important;
    font-weight: 500;
}

/* Hide radio circles */
[data-testid="stSidebar"] .stRadio label div:first-child {
    display: none;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)




st.title("Employee Attrition Prediction")

# Project branding

# ==========================================
# ATTRITION AI SIDEBAR
# ==========================================
st.sidebar.markdown("""
<div style="background-color:#19263B; padding:15px; border-radius:12px;">
    <h2 style="color:white; margin:0;">👥 AttritionAI</h2>
    <p style="color:#94A3B8; font-size:10px; letter-spacing:1px;">
        WORKFORCE INTELLIGENCE
    </p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("### MAIN MENU")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Model Comparison", "EDA Insights", "About Project"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="background-color:#19263B; padding:15px; border-radius:12px;">
    <h4 style="color:#60A5FA;">🧠 AI MODELS ACTIVE</h4>
    <p style="color:white;">
        ✓ Decision Tree<br>
        ✓ Random Forest<br>
        ✓ SHAP Explainability<br>
        ✓ LIME Explainability
    </p>
    <p style="color:#22C55E;">● PROJECT DASHBOARD READY</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Explainable AI using SHAP and LIME")

if page == "Home":
   
    st.markdown("""
    <div style="
       background: linear-gradient(135deg, #0f172a, #1e293b);
border: 1px solid #334155;
        padding: 35px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
    ">
        <h1 style="color: white;">
            Welcome to Employee Attrition AI
        </h1>
        <h3 style="color: #dbeafe;">
            Predict. Understand. Improve.
        </h3>
        <p style="font-size: 16px;">
            An intelligent employee attrition prediction
            system powered by Machine Learning,
            SHAP and LIME.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Project Statistics")

    total_employees = len(df)
    employees_left = (df["Attrition"] == "Yes").sum()
    attrition_rate = (employees_left / total_employees) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Employees", total_employees)
    col2.metric("Employees Left", employees_left)
    col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

    st.divider()

    st.subheader("What can you explore?")

    st.markdown("""
    - **Prediction:** Predict employee attrition.
    - **Model Comparison:** Compare Decision Tree and Random Forest.
    - **EDA Insights:** Explore employee data through charts.
    - **Explainable AI:** Understand predictions using SHAP and LIME.
    - **About Project:** Learn about the project and its workflow.
    """)

    st.success("Explore the different sections using the sidebar.")

    st.stop()



if page == "Model Comparison":
    st.title("Model Performance Comparison")

    st.write(
        "Comparison of Decision Tree and Random Forest "
        "using the same unseen test dataset."
    )

    # Generate predictions
    dt_predictions = decision_tree_model.predict(X_test)
    rf_predictions = random_forest_model.predict(X_test)

    # Calculate evaluation metrics
    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Attrition Precision",
            "Attrition Recall",
            "Attrition F1-score"
        ],
        "Decision Tree": [
            accuracy_score(y_test, dt_predictions) * 100,
            precision_score(y_test, dt_predictions, zero_division=0) * 100,
            recall_score(y_test, dt_predictions, zero_division=0) * 100,
            f1_score(y_test, dt_predictions, zero_division=0) * 100
        ],
        "Random Forest": [
            accuracy_score(y_test, rf_predictions) * 100,
            precision_score(y_test, rf_predictions, zero_division=0) * 100,
            recall_score(y_test, rf_predictions, zero_division=0) * 100,
            f1_score(y_test, rf_predictions, zero_division=0) * 100
        ]
    })

    # Display metrics
    st.subheader("Evaluation Metrics")

    display_comparison = comparison.copy()

    for column in ["Decision Tree", "Random Forest"]:
        display_comparison[column] = display_comparison[column].map(
            lambda value: f"{value:.2f}%"
        )

    st.dataframe(
        display_comparison,
        hide_index=True,
        use_container_width=True
    )

    # Performance chart
    st.subheader("Model Performance Visualization")

    chart_data = comparison.set_index("Metric")
    st.bar_chart(chart_data)

    # Confusion matrices
    st.subheader("Confusion Matrix Comparison")

    dt_cm = confusion_matrix(y_test, dt_predictions, labels=[0, 1])
    rf_cm = confusion_matrix(y_test, rf_predictions, labels=[0, 1])
    st.subheader("Confusion Matrix Heatmaps")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Decision Tree")

        fig, ax = plt.subplots()
        sns.heatmap(
            dt_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Stay", "Leave"],
            yticklabels=["Stay", "Leave"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.write("Random Forest")

        fig, ax = plt.subplots()
        sns.heatmap(
            rf_cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=["Stay", "Leave"],
            yticklabels=["Stay", "Leave"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    decision_tree_cm = pd.DataFrame(
        dt_cm,
        index=["Actual Stayed", "Actual Left"],
        columns=["Predicted Stayed", "Predicted Left"]
    )

    random_forest_cm = pd.DataFrame(
        rf_cm,
        index=["Actual Stayed", "Actual Left"],
        columns=["Predicted Stayed", "Predicted Left"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("Decision Tree")
        st.dataframe(decision_tree_cm)

    with col2:
        st.write("Random Forest")
        st.dataframe(random_forest_cm)

    # Classification reports
    st.subheader("Classification Report")

    dt_report = classification_report(
        y_test,
        dt_predictions,
        labels=[0, 1],
        target_names=["Stayed", "Left"],
        output_dict=True,
        zero_division=0
    )

    rf_report = classification_report(
        y_test,
        rf_predictions,
        labels=[0, 1],
        target_names=["Stayed", "Left"],
        output_dict=True,
        zero_division=0
    )

    report = pd.DataFrame({
        "Class": ["Stayed", "Left"],
        "Decision Tree Precision": [
            dt_report["Stayed"]["precision"] * 100,
            dt_report["Left"]["precision"] * 100
        ],
        "Decision Tree Recall": [
            dt_report["Stayed"]["recall"] * 100,
            dt_report["Left"]["recall"] * 100
        ],
        "Decision Tree F1-score": [
            dt_report["Stayed"]["f1-score"] * 100,
            dt_report["Left"]["f1-score"] * 100
        ],
        "Random Forest Precision": [
            rf_report["Stayed"]["precision"] * 100,
            rf_report["Left"]["precision"] * 100
        ],
        "Random Forest Recall": [
            rf_report["Stayed"]["recall"] * 100,
            rf_report["Left"]["recall"] * 100
        ],
        "Random Forest F1-score": [
            rf_report["Stayed"]["f1-score"] * 100,
            rf_report["Left"]["f1-score"] * 100
        ]
    })

    st.dataframe(
        report.style.format({
            column: "{:.2f}%"
            for column in report.columns
            if column != "Class"
        }),
        hide_index=True,
        use_container_width=True
    )

    # Accuracy comparison
    st.subheader("Accuracy Comparison")

    col1, col2 = st.columns(2)

    dt_accuracy = accuracy_score(y_test, dt_predictions) * 100
    rf_accuracy = accuracy_score(y_test, rf_predictions) * 100

    col1.metric("Decision Tree", f"{dt_accuracy:.2f}%")
    col2.metric("Random Forest", f"{rf_accuracy:.2f}%")

    # Important observations
    st.subheader("Important Observations")

    dt_recall = recall_score(
        y_test, dt_predictions, zero_division=0
    ) * 100

    rf_recall = recall_score(
        y_test, rf_predictions, zero_division=0
    ) * 100

    if dt_accuracy > rf_accuracy:
        st.write("Decision Tree achieved higher accuracy.")
    elif rf_accuracy > dt_accuracy:
        st.write("Random Forest achieved higher accuracy.")
    else:
        st.write("Both models achieved the same accuracy.")

    if dt_recall > rf_recall:
        st.write("Decision Tree identified more actual attrition cases.")
    elif rf_recall > dt_recall:
        st.write("Random Forest identified more actual attrition cases.")
    else:
        st.write("Both models achieved the same attrition recall.")

    st.info(
        "Attrition recall is important because it measures how many "
        "employees who actually left were correctly identified by "
        "the model. These predictions should support, not replace, "
        "human HR review."
    )

    st.stop()


if page == "EDA Insights":
    st.title("Exploratory Data Analysis")
    st.write("Explore employee attrition patterns through interactive charts.")

    # Department Filter
    st.subheader("Filter Employee Data")

    departments = ["All Departments"] + sorted(df["Department"].unique().tolist())

    selected_department = st.selectbox(
        "Select Department",
        departments
    )

    if selected_department == "All Departments":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["Department"] == selected_department]

    # KPI Cards
    total_employees = len(filtered_df)
    employees_left = (filtered_df["Attrition"] == "Yes").sum()
    attrition_rate = (employees_left / total_employees) * 100 if total_employees > 0 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Employees", total_employees)
    col2.metric("Employees Left", employees_left)
    col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

    # Chart 1: Attrition Distribution
    st.subheader("1. Employee Attrition Distribution")
    attrition_counts = filtered_df["Attrition"].value_counts()
    st.bar_chart(attrition_counts)

    # Chart 2: Attrition by Overtime
    st.subheader("2. Attrition Rate by Overtime")
    overtime_rate = filtered_df.groupby("OverTime")["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    st.bar_chart(overtime_rate)

    # Chart 3: Attrition by Department
    st.subheader("3. Attrition Rate by Department")
    department_rate = filtered_df.groupby("Department")["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    st.bar_chart(department_rate)

    # Chart 4: Attrition by Age Group
    st.subheader("4. Attrition Rate by Age Group")

    filtered_df = filtered_df.copy()
    filtered_df["AgeGroup"] = pd.cut(
        filtered_df["Age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65"]
    )

    age_rate = filtered_df.groupby("AgeGroup", observed=False)["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    st.bar_chart(age_rate)

    st.stop()


if page == "About Project":
    st.title("Project Overview")

    st.write("""
    ## Employee Attrition Prediction
    ### Explainable AI using SHAP and LIME

    This project uses Machine Learning to predict whether
    an employee may leave an organization and explains
    the factors influencing the model's prediction.
    """)

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Predict employee attrition using Machine Learning.
    - Identify patterns in employee data.
    - Compare Decision Tree and Random Forest models.
    - Explain predictions using SHAP and LIME.
    - Present results through an interactive dashboard.
    """)

    st.subheader("Technologies Used")

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Models", "2")
    col2.metric("XAI Techniques", "2")
    col3.metric("Dataset Records", "1,470")

    st.write("""
    **Programming:** Python

    **Libraries:** Pandas, NumPy, Scikit-learn

    **Explainable AI:** SHAP, LIME

    **Dashboard:** Streamlit

    **Dataset:** IBM HR Analytics Employee Attrition Dataset
    """)

    st.divider()

    st.subheader("Project Workflow")

    st.markdown("""
    1. Data Collection
    2. Data Cleaning and Preprocessing
    3. Exploratory Data Analysis
    4. Machine Learning Model Training
    5. Model Evaluation
    6. SHAP and LIME Explanations
    7. Streamlit Dashboard Development
    """)
    st.divider()
    st.subheader("Key Findings")

    # Calculate predictions using test data
    dt_predictions = decision_tree_model.predict(X_test)
    rf_predictions = random_forest_model.predict(X_test)

    # Calculate actual metrics
    dt_accuracy = accuracy_score(y_test, dt_predictions) * 100
    rf_accuracy = accuracy_score(y_test, rf_predictions) * 100

    dt_recall = recall_score(y_test, dt_predictions, zero_division=0) * 100
    rf_recall = recall_score(y_test, rf_predictions, zero_division=0) * 100

    st.markdown(f"""
    - Decision Tree accuracy: {dt_accuracy:.2f}%
    - Random Forest accuracy: {rf_accuracy:.2f}%
    - Decision Tree attrition recall: {dt_recall:.2f}%
    - Random Forest attrition recall: {rf_recall:.2f}%
    """)

    st.info(
        "The predictions are intended to support HR analysis. "
        "They should not be treated as proof that an employee "
        "will leave an organization."
    )

    st.success("Employee Attrition Prediction Project")
    st.stop()


st.write(
    "Enter employee details below to predict "
    "whether the employee is likely to leave."
)
st.subheader("Choose Machine Learning Model")

model_choice = st.selectbox(
    "Select Model",
    ["Decision Tree", "Random Forest"]
)

if model_choice == "Decision Tree":
    model = decision_tree_model
else:
    model = random_forest_model
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
input_data = {}

with st.form("employee_form"):

    st.subheader("Employee Information")

    columns = st.columns(2)

    for i, col in enumerate(employee_data.columns):

        with columns[i % 2]:

            if col in categorical_columns:

                categories = sorted(
                    employee_data[col].dropna().unique().tolist()
                )

                input_data[col] = st.selectbox(
                    col,
                    categories
                )

            else:

                minimum = int(employee_data[col].min())
                maximum = int(employee_data[col].max())
                default = int(employee_data[col].median())

                input_data[col] = st.number_input(
                    col,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1
                )

    submitted = st.form_submit_button(
        "Predict Attrition"
    )
if submitted:
    st.session_state["show_prediction"] = True
    st.session_state.pop("retention_scenario", None)

    # Save employee details and selected model
    st.session_state["saved_input_data"] = input_data.copy()
    st.session_state["saved_model_choice"] = model_choice

if submitted or st.session_state.get("show_prediction", False):

    # Restore saved employee details and model
    input_data = st.session_state.get("saved_input_data", input_data)
    model_choice = st.session_state.get("saved_model_choice", model_choice)

    if model_choice == "Decision Tree":
        model = decision_tree_model
    else:
        model = random_forest_model

    input_df = pd.DataFrame([input_data])

    # Preserve the original category levels
    for col in categorical_columns:
        categories = sorted(
            employee_data[col].dropna().unique().tolist()
        )
        input_df[col] = pd.Categorical(
            input_df[col],
            categories=categories
        )

    # Convert categorical values into numerical features
    input_encoded = pd.get_dummies(
        input_df,
        drop_first=True,
        dtype=int
    )

    # Match the training feature columns
    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]

    result = "Likely to Leave" if prediction == 1 else "Likely to Stay"

    if submitted:
        st.session_state.prediction_history.append({
            "Model": model_choice,
            "Prediction": result,
            "Probability of Leaving": f"{probability[1] * 100:.2f}%",
            "Date": pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")
    })
    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("Employee is predicted as likely to leave.")
    else:
        st.success("Employee is predicted as likely to stay.")

    st.metric("Probability of Leaving", f"{probability[1] * 100:.2f}%")
    st.progress(float(probability[1]))

    st.metric("Probability of Staying", f"{probability[0] * 100:.2f}%")
    st.progress(float(probability[0]))

    st.subheader("Why did the model make this prediction?")
    st.subheader("SHAP Explanation")
    st.write("Which factors influenced this employee's prediction?")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(input_encoded)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        employee_shap = shap_values[1][0]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        employee_shap = shap_values[0, :, 1]

    elif isinstance(shap_values, np.ndarray):
        employee_shap = shap_values[0]

    else:
        employee_shap = shap_values.values[0]

    # Create SHAP importance table
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": employee_shap
    })

    shap_importance["Absolute Value"] = (
        shap_importance["SHAP Value"].abs()
    )

    shap_importance = shap_importance.sort_values(
        by="Absolute Value",
        ascending=False
    ).head(10)

    # Separate factors based on SHAP values
    leaving_factors = shap_importance[
        shap_importance["SHAP Value"] > 0
    ]

    staying_factors = shap_importance[
        shap_importance["SHAP Value"] < 0
    ]

    # Factors supporting Leaving
    st.subheader("Factors Supporting Leaving")

    if not leaving_factors.empty:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.barh(
            leaving_factors["Feature"],
            leaving_factors["SHAP Value"]
        )

        ax.invert_yaxis()
        ax.set_xlabel("SHAP Value")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("No factors pushed the prediction towards Leaving.")

    # Factors supporting Staying
    st.subheader("Factors Supporting Staying")

    if not staying_factors.empty:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.barh(
            staying_factors["Feature"],
            staying_factors["SHAP Value"]
        )

        ax.invert_yaxis()
        ax.set_xlabel("SHAP Value")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("No factors pushed the prediction towards Staying.")
        
    # LIME Explanation
    st.subheader("LIME Explanation")
    st.write("Why was this individual employee predicted this way?")

    lime_explanation = lime_explainer.explain_instance(
        data_row=input_encoded.iloc[0].values,
        predict_fn=model.predict_proba,
        num_features=10
    )

    st.write(lime_explanation.as_list())

    lime_html = lime_explanation.as_html()
    lime_html = lime_html.replace(
        "<body>",
        '<body style="background-color: white; color: black;">'
    )

    components.html(
        lime_html,
        height=700,
        scrolling=True
    )
    
    # ==========================================
    # EMPLOYEE EXIT IMPACT ANALYSIS
    # ==========================================

    st.divider()
    st.subheader("Employee Exit Impact Analysis")

    st.write(
        "Review employee information to identify areas "
        "that may require attention during a transition."
    )

    # Extract employee information
    job_role = input_data["JobRole"]
    job_level = input_data["JobLevel"]
    years_company = input_data["YearsAtCompany"]
    years_role = input_data["YearsInCurrentRole"]
    total_experience = input_data["TotalWorkingYears"]
    training = input_data["TrainingTimesLastYear"]

    st.write("### Employee Profile")

    col1, col2, col3 = st.columns(3)

    col1.metric("Job Role", job_role)
    col2.metric("Job Level", job_level)
    col3.metric("Years at Company", years_company)

    col1, col2, col3 = st.columns(3)

    col1.metric("Years in Current Role", years_role)
    col2.metric("Total Experience", total_experience)
    col3.metric("Training Last Year", training)

    st.write("### Transition Planning Indicators")

    indicators = []

    if job_level >= 4:
        indicators.append({
            "Indicator": "Senior job level",
            "Observation": "Review responsibilities and handover requirements."
        })

    if years_company >= employee_data["YearsAtCompany"].median():
        indicators.append({
            "Indicator": "Longer organizational tenure",
            "Observation": "Review accumulated knowledge and ongoing responsibilities."
        })

    if years_role >= employee_data["YearsInCurrentRole"].median():
        indicators.append({
            "Indicator": "Longer time in current role",
            "Observation": "Identify role-specific tasks that may need documentation."
        })

    if training <= employee_data["TrainingTimesLastYear"].median():
        indicators.append({
            "Indicator": "Training activity",
            "Observation": "Check whether another team member needs additional training."
        })

    if indicators:
        indicator_df = pd.DataFrame(indicators)
        st.dataframe(
            indicator_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success(
            "No specific transition-planning indicators "
            "were identified from these selected fields."
        )

    st.write("### HR Review Checklist")

    st.markdown("""
    - Review the employee's key responsibilities.
    - Identify tasks that require specialized knowledge.
    - Check whether another employee can handle those tasks.
    - Prepare documentation and knowledge-transfer plans.
    - Estimate recruitment and training requirements separately.
    """)

    st.info(
        "This assessment uses employee profile indicators as "
        "prompts for HR review. It does not measure actual "
        "business loss, employee value, or replacement difficulty. "
        "A human HR review is required."
    )

    # ==========================================
    # AI EMPLOYEE RETENTION RECOMMENDATIONS
    # ==========================================

    st.divider()
    st.subheader("AI Employee Retention Recommendations")

    st.write(
        "Personalized suggestions based on the employee's "
        "workplace information."
    )

    recommendations = []

    if input_data["OverTime"] == "Yes":
        recommendations.append({
            "Area": "Work-life balance",
            "Observation": "Employee works overtime.",
            "Recommendation": "Review workload, overtime frequency and staffing requirements."
        })

    if input_data["JobSatisfaction"] <= 2:
        recommendations.append({
            "Area": "Job satisfaction",
            "Observation": "Job satisfaction is relatively low.",
            "Recommendation": "Discuss workplace concerns and explore suitable role improvements."
        })

    if input_data["EnvironmentSatisfaction"] <= 2:
        recommendations.append({
            "Area": "Work environment",
            "Observation": "Environment satisfaction is relatively low.",
            "Recommendation": "Collect feedback about the work environment and identify improvements."
        })

    if input_data["WorkLifeBalance"] <= 2:
        recommendations.append({
            "Area": "Work-life balance",
            "Observation": "Work-life balance rating is relatively low.",
            "Recommendation": "Consider flexible working arrangements and workload adjustments."
        })

    if input_data["RelationshipSatisfaction"] <= 2:
        recommendations.append({
            "Area": "Workplace relationships",
            "Observation": "Relationship satisfaction is relatively low.",
            "Recommendation": "Encourage constructive feedback, mentoring and team communication."
        })

    if input_data["YearsSinceLastPromotion"] >= employee_data["YearsSinceLastPromotion"].median():
        recommendations.append({
            "Area": "Career growth",
            "Observation": "Time since last promotion is relatively high.",
            "Recommendation": "Discuss career goals, promotion criteria and development opportunities."
        })

    if input_data["TrainingTimesLastYear"] <= employee_data["TrainingTimesLastYear"].median():
        recommendations.append({
            "Area": "Learning and development",
            "Observation": "Training activity is relatively low.",
            "Recommendation": "Explore relevant training programs and skill-development opportunities."
        })

    if recommendations:
        recommendation_df = pd.DataFrame(recommendations)

        st.dataframe(
            recommendation_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success(
            "No specific improvement areas were identified "
            "by these simple rules. Continue regular employee feedback."
        )

    st.info(
        "These recommendations are rule-based suggestions, "
        "not guaranteed solutions. HR should discuss them "
        "with the employee before taking action."
    )
    
    # ==========================================
    # WHAT-IF RETENTION SIMULATOR
    # ==========================================

    st.divider()
    
    st.subheader("What-if Retention Simulator")

    st.write(
        "Explore how changing selected employee factors "
        "affects the model's attrition prediction."
    )

    with st.form("retention_simulator_form"):

        st.write("### Adjust Employee Factors")

        scenario_overtime = st.selectbox(
            "Overtime",
            ["No", "Yes"],
            index=["No", "Yes"].index(input_data["OverTime"])
        )

        scenario_job_satisfaction = st.slider(
            "Job Satisfaction", 1, 4,
            int(input_data["JobSatisfaction"])
        )

        scenario_environment = st.slider(
            "Environment Satisfaction", 1, 4,
            int(input_data["EnvironmentSatisfaction"])
        )

        scenario_worklife = st.slider(
            "Work-Life Balance", 1, 4,
            int(input_data["WorkLifeBalance"])
        )

        scenario_training = st.slider(
            "Training Times Last Year",
            int(employee_data["TrainingTimesLastYear"].min()),
            int(employee_data["TrainingTimesLastYear"].max()),
            int(input_data["TrainingTimesLastYear"])
        )

        simulate = st.form_submit_button(
            "Simulate Retention Scenario"
        )

    if simulate:

        scenario_data = input_data.copy()

        scenario_data["OverTime"] = scenario_overtime
        scenario_data["JobSatisfaction"] = scenario_job_satisfaction
        scenario_data["EnvironmentSatisfaction"] = scenario_environment
        scenario_data["WorkLifeBalance"] = scenario_worklife
        scenario_data["TrainingTimesLastYear"] = scenario_training

        scenario_df = pd.DataFrame([scenario_data])

        for col in categorical_columns:
            categories = sorted(
                employee_data[col].dropna().unique().tolist()
            )
            scenario_df[col] = pd.Categorical(
                scenario_df[col],
                categories=categories
            )

        scenario_encoded = pd.get_dummies(
            scenario_df,
            drop_first=True,
            dtype=int
        )

        scenario_encoded = scenario_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        scenario_prediction = model.predict(scenario_encoded)[0]

        scenario_probability = model.predict_proba(
            scenario_encoded
        )[0]

        st.session_state["retention_scenario"] = {
            "original_risk": float(probability[1] * 100),
            "scenario_risk": float(scenario_probability[1] * 100),
            "prediction": int(scenario_prediction)
        }

    # Display saved results
    if "retention_scenario" in st.session_state:

        saved = st.session_state["retention_scenario"]

        st.write("### Scenario Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Original Attrition Risk",
            f"{saved['original_risk']:.2f}%"
        )

        col2.metric(
            "Scenario Attrition Risk",
            f"{saved['scenario_risk']:.2f}%",
            delta=f"{saved['scenario_risk'] - saved['original_risk']:+.2f}%"
        )

        st.write("### Scenario Prediction")

        if saved["prediction"] == 1:
            st.warning("Scenario result: Likely to Leave")
        else:
            st.success("Scenario result: Likely to Stay")

        st.info(
            "This is a model-based scenario, not proof that "
            "changing these factors will cause an employee "
            "to stay or leave. Actual outcomes may differ."
        )
            # Download Prediction Report
    from io import BytesIO
    from datetime import datetime
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )

    st.subheader("Download Prediction Report")

    # Create PDF in memory
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#243B64"),
        fontSize=22,
        spaceAfter=8
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#243B64"),
        spaceBefore=14,
        spaceAfter=8
    )

    normal_style = styles["BodyText"]

    elements = []

    # Report title
    elements.append(Paragraph("AttritionAI", title_style))
    elements.append(
        Paragraph("Employee Attrition Prediction Report", styles["Heading2"])
    )
    elements.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}",
            normal_style
        )
    )
    elements.append(Spacer(1, 15))

    # Prediction summary
    elements.append(Paragraph("1. Prediction Summary", heading_style))

    prediction_data = [
        ["Factor", "Result"],
        ["Selected Model", str(model_choice)],
        ["Prediction", str(result)],
        ["Probability of Leaving", f"{probability[1] * 100:.2f}%"],
        ["Probability of Staying", f"{probability[0] * 100:.2f}%"]
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[2.5 * inch, 3.8 * inch]
    )

    prediction_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B64")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(prediction_table)

    # Employee details
    elements.append(Paragraph("2. Employee Information", heading_style))

    employee_rows = [["Employee Feature", "Value"]]

    for feature, value in input_data.items():
        employee_rows.append([
            str(feature),
            str(value)
        ])

    employee_table = Table(
        employee_rows,
        colWidths=[2.5 * inch, 3.8 * inch],
        repeatRows=1
    )

    employee_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B64")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(employee_table)

    # Exit impact analysis
    elements.append(Paragraph("3. Employee Exit Impact Analysis", heading_style))

    impact_fields = [
        "JobRole",
        "JobLevel",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "TotalWorkingYears",
        "TrainingTimesLastYear"
    ]

    impact_rows = [["Indicator", "Employee Information"]]

    for field in impact_fields:
        if field in input_data:
            impact_rows.append([
                field,
                str(input_data[field])
            ])

    impact_table = Table(
        impact_rows,
        colWidths=[2.5 * inch, 3.8 * inch]
    )

    impact_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B64")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    elements.append(impact_table)

    elements.append(Paragraph(
        "These details can help HR plan knowledge transfer, "
        "role continuity and employee transition support.",
        normal_style
    ))

    # Retention recommendations
    elements.append(Paragraph("4. Retention Recommendations", heading_style))

    recommendations = []

    if input_data.get("OverTime") == "Yes":
        recommendations.append(
            "Review overtime levels and consider a more balanced workload."
        )

    if input_data.get("JobSatisfaction", 5) <= 2:
        recommendations.append(
            "Discuss job satisfaction and identify possible workplace concerns."
        )

    if input_data.get("EnvironmentSatisfaction", 5) <= 2:
        recommendations.append(
            "Review the employee's work environment and support needs."
        )

    if input_data.get("WorkLifeBalance", 5) <= 2:
        recommendations.append(
            "Explore flexible working arrangements and work-life balance support."
        )

    if input_data.get("RelationshipSatisfaction", 5) <= 2:
        recommendations.append(
            "Encourage regular feedback and improve team relationships."
        )

    if input_data.get("YearsSinceLastPromotion", 0) >= 3:
        recommendations.append(
            "Review career growth opportunities and promotion pathways."
        )

    if input_data.get("TrainingTimesLastYear", 0) <= 1:
        recommendations.append(
            "Consider additional training and professional development."
        )

    if not recommendations:
        recommendations.append(
            "Continue regular employee feedback, career development "
            "and workplace support."
        )

    for recommendation in recommendations:
        elements.append(
            Paragraph("• " + escape(recommendation), normal_style)
        )
        elements.append(Spacer(1, 5))

    # What-if scenario results, if available
    elements.append(Paragraph("5. What-if Retention Analysis", heading_style))

    if "retention_scenario" in st.session_state:
        saved = st.session_state["retention_scenario"]

        scenario_rows = [
            ["Scenario Metric", "Result"],
            ["Original Attrition Risk", f"{saved['original_risk']:.2f}%"],
            ["Scenario Attrition Risk", f"{saved['scenario_risk']:.2f}%"],
            [
                "Scenario Prediction",
                "Likely to Leave" if saved["prediction"] == 1
                else "Likely to Stay"
            ]
        ]

        scenario_table = Table(
            scenario_rows,
            colWidths=[2.5 * inch, 3.8 * inch]
        )

        scenario_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B64")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ]))

        elements.append(scenario_table)

        elements.append(Spacer(1, 8))
        elements.append(Paragraph(
            "Note: What-if results are model-based estimates. "
            "They do not prove that changing these factors will "
            "cause an employee to stay or leave.",
            normal_style
        ))

    else:
        elements.append(Paragraph(
            "No what-if scenario has been generated for this prediction.",
            normal_style
        ))

    # Disclaimer
    elements.append(Spacer(1, 18))
    elements.append(Paragraph("Disclaimer", heading_style))
    elements.append(Paragraph(
        "This report provides machine-learning-based estimates "
        "for decision support only. Predictions should not be used "
        "as the sole basis for employment decisions. HR should "
        "consider employee feedback and additional context.",
        normal_style
    ))

    # Generate PDF
    doc.build(elements)

    pdf_data = pdf_buffer.getvalue()

    st.download_button(
        label="Download Employee Prediction Report (PDF)",
        data=pdf_data,
        file_name="AttritionAI_Employee_Report.pdf",
        mime="application/pdf"
    )
    st.subheader("Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No predictions made yet.")
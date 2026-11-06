from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
X_train, _, y_train, _ = train_test_split(
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
        "on the same test dataset."
    )

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Attrition Precision", "Attrition Recall", "Attrition F1-score"],
        "Decision Tree": [77.21, 36, 53, 43],
        "Random Forest": [83.33, 47, 32, 38]
    })

    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.subheader("Model Performance Visualization")

    chart_data = comparison.set_index("Metric")

    st.bar_chart(chart_data)
    st.subheader("Confusion Matrix Comparison")

    col1, col2 = st.columns(2)

    decision_tree_cm = pd.DataFrame(
        [[202, 45], [22, 25]],
        index=["Actual Stayed", "Actual Left"],
        columns=["Predicted Stayed", "Predicted Left"]
    )

    random_forest_cm = pd.DataFrame(
        [[230, 17], [32, 15]],
        index=["Actual Stayed", "Actual Left"],
        columns=["Predicted Stayed", "Predicted Left"]
    )

    with col1:
        st.write("Decision Tree")
        st.dataframe(decision_tree_cm)

    with col2:
        st.write("Random Forest")
        st.dataframe(random_forest_cm)
        
    st.subheader("Classification Report")

    report = pd.DataFrame({
        "Class": ["Stayed", "Left"],
        "Decision Tree Precision": [90, 36],
        "Decision Tree Recall": [82, 53],
        "Decision Tree F1-score": [86, 43],
        "Random Forest Precision": [88, 47],
        "Random Forest Recall": [93, 32],
        "Random Forest F1-score": [90, 38]
    })

    st.dataframe(report, hide_index=True, use_container_width=True)

    st.subheader("Accuracy Comparison")
    col1, col2 = st.columns(2)
    col1.metric("Decision Tree", "77.21%")
    col2.metric("Random Forest", "83.33%")

    st.subheader("Important Observation")
    st.write("""
    - Random Forest achieved higher accuracy.
    - Random Forest achieved higher attrition precision.
    - Decision Tree achieved higher attrition recall and F1-score.
    """)

    st.info(
        "For employee attrition prediction, recall is important "
        "because missing employees who may leave can be costly."
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

    st.markdown("""
    - Decision Tree accuracy: 77.21%
    - Random Forest accuracy: 83.33%
    - Decision Tree attrition recall: 53%
    - Random Forest attrition recall: 32%
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

    # Download Prediction Report
    st.subheader("Download Prediction Report")

    report_data = [
        {
            "Section": "Prediction",
            "Factor": "Selected Model",
            "Value": model_choice
        },
        {
            "Section": "Prediction",
            "Factor": "Result",
            "Value": result
        },
        {
            "Section": "Prediction",
            "Factor": "Probability of Leaving",
            "Value": f"{probability[1] * 100:.2f}%"
        },
        {
            "Section": "Prediction",
            "Factor": "Probability of Staying",
            "Value": f"{probability[0] * 100:.2f}%"
        }
    ]

    # Add SHAP results
    for _, row in shap_importance.iterrows():
        report_data.append({
            "Section": "SHAP",
            "Factor": row["Feature"],
            "Value": row["SHAP Value"]
        })

    # Add LIME results
    for feature, value in lime_explanation.as_list():
        report_data.append({
            "Section": "LIME",
            "Factor": feature,
            "Value": value
        })

    report_df = pd.DataFrame(report_data)
    csv_report = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Employee Prediction Report",
        data=csv_report,
        file_name="employee_prediction_report.csv",
        mime="text/csv"
    )
    st.subheader("Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No predictions made yet.")
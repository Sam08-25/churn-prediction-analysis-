import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    layout="wide",
    page_icon="🔮"
)

# ---------------- LOAD FILES ---------------- #

cust_model = joblib.load('models/customer_churn_lgbm.pkl')
emp_model  = joblib.load('models/employee_churn_lgbm.pkl')

cust_data = pd.read_csv(
    'data/processed/customer_churn_lgbm_predictions.csv'
)

emp_data = pd.read_csv(
    'data/processed/employee_churn_lgbm_predictions.csv'
)

# ---------------- SIDEBAR ---------------- #

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Customer churn",
        "Employee attrition",
        "Risk simulator"
    ]
)

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.title("Dual Churn Prediction Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    cust_hr = (cust_data['churn_probability'] >= 0.7).sum()
    emp_hr  = (emp_data['churn_probability'] >= 0.7).sum()

    col1.metric("High-risk customers", cust_hr)

    col2.metric("High-risk employees", emp_hr)

    col3.metric(
        "Customer churn rate",
        f"{cust_data['actual'].mean()*100:.1f}%"
    )

    col4.metric(
        "Employee attrition rate",
        f"{emp_data['actual'].mean()*100:.1f}%"
    )

    st.subheader("Churn Probability Distributions")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots()

        ax.hist(
            cust_data['churn_probability'],
            bins=30,
            color='#534AB7'
        )

        ax.set_title('Customer Churn Probability')

        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots()

        ax.hist(
            emp_data['churn_probability'],
            bins=30,
            color='#1D9E75'
        )

        ax.set_title('Employee Attrition Probability')

        st.pyplot(fig)

# =========================================================
# CUSTOMER CHURN PAGE
# =========================================================

elif page == "Customer churn":

    st.title("Customer Churn Analysis")

    st.subheader("Top High-Risk Customers")

    high_risk = cust_data[
        cust_data['churn_probability'] >= 0.7
    ]

    st.dataframe(
        high_risk.head(20)
    )

    st.subheader("Customer Churn Probability Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        cust_data['churn_probability'],
        bins=30,
        color='#534AB7'
    )

    ax.set_xlabel("Probability")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    st.subheader("Top 10 Highest Risk Customers")

    top_customers = cust_data.sort_values(
        'churn_probability',
        ascending=False
    ).head(10)

    st.dataframe(top_customers)

# =========================================================
# EMPLOYEE ATTRITION PAGE
# =========================================================

elif page == "Employee attrition":

    st.title("Employee Attrition Analysis")

    st.subheader("Top High-Risk Employees")

    high_risk = emp_data[
        emp_data['churn_probability'] >= 0.7
    ]

    st.dataframe(
        high_risk.head(20)
    )

    st.subheader("Employee Attrition Probability Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        emp_data['churn_probability'],
        bins=30,
        color='#1D9E75'
    )

    ax.set_xlabel("Probability")
    ax.set_ylabel("Employees")

    st.pyplot(fig)

    st.subheader("Top 10 Highest Risk Employees")

    top_employees = emp_data.sort_values(
        'churn_probability',
        ascending=False
    ).head(10)

    st.dataframe(top_employees)

# =========================================================
# RISK SIMULATOR
# =========================================================

elif page == "Risk simulator":

    st.title("Risk Simulator")

    st.write(
        "Enter profile information to predict churn risk."
    )

    mode = st.radio(
        "Predict for:",
        ["Customer", "Employee"]
    )

    # ---------------- CUSTOMER SIMULATOR ---------------- #

    if mode == "Customer":

        col1, col2 = st.columns(2)

        tenure = col1.slider(
            "Tenure (months)",
            0,
            72,
            12
        )

        monthly = col2.slider(
            "Monthly Charges ($)",
            18,
            120,
            65
        )

        contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        if st.button("Predict Customer Churn Risk"):

            sample = cust_data.drop(
                ['actual', 'predicted', 'churn_probability'],
                axis=1
            ).iloc[[0]].copy()

            sample['tenure'] = tenure
            sample['MonthlyCharges'] = monthly

            prob = cust_model.predict_proba(sample)[0][1]

            if prob > 0.7:
                tier = "HIGH RISK"
                color = "red"

            elif prob > 0.4:
                tier = "MEDIUM RISK"
                color = "orange"

            else:
                tier = "LOW RISK"
                color = "green"

            st.markdown(
                f"## Churn Probability: {prob*100:.1f}%"
            )

            st.markdown(
                f"### Risk Level: :{color}[{tier}]"
            )

            # SHAP explanation

            explainer = shap.TreeExplainer(cust_model)

            sv = explainer.shap_values(sample)

            top3 = pd.Series(
                np.abs(sv[0]),
                index=sample.columns
            ).nlargest(3)

            st.subheader("Top Risk Factors")

            for feat, val in top3.items():

                st.write(
                    f"• {feat} → impact score {val:.3f}"
                )

    # ---------------- EMPLOYEE SIMULATOR ---------------- #

    else:

        col1, col2 = st.columns(2)

        monthly_income = col1.slider(
            "Monthly Income",
            1000,
            20000,
            5000
        )

        overtime = col2.selectbox(
            "Overtime",
            ["Yes", "No"]
        )

        years = st.slider(
            "Years at Company",
            0,
            40,
            5
        )

        if st.button("Predict Employee Attrition Risk"):

            sample = emp_data.drop(
                ['actual', 'predicted', 'churn_probability'],
                axis=1
            ).iloc[[0]].copy()

            if 'MonthlyIncome' in sample.columns:
                sample['MonthlyIncome'] = monthly_income

            if 'YearsAtCompany' in sample.columns:
                sample['YearsAtCompany'] = years

            prob = emp_model.predict_proba(sample)[0][1]

            if prob > 0.7:
                tier = "HIGH RISK"
                color = "red"

            elif prob > 0.4:
                tier = "MEDIUM RISK"
                color = "orange"

            else:
                tier = "LOW RISK"
                color = "green"

            st.markdown(
                f"## Attrition Probability: {prob*100:.1f}%"
            )

            st.markdown(
                f"### Risk Level: :{color}[{tier}]"
            )

            explainer = shap.TreeExplainer(emp_model)

            sv = explainer.shap_values(sample)

            top3 = pd.Series(
                np.abs(sv[0]),
                index=sample.columns
            ).nlargest(3)

            st.subheader("Top Risk Factors")

            for feat, val in top3.items():

                st.write(
                    f"• {feat} → impact score {val:.3f}"
                )
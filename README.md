# AI-Driven Credit Risk Assessment Engine

## Project Overview
This project focuses on building a high-precision **Credit Risk Scoring System** for digital lending environments. Using the **XGBoost** algorithm and **SMOTE** for handling class imbalance, this engine predicts the probability of loan default with high interpretability, helping financial institutions mitigate risks while maintaining operational efficiency.

🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/anisa04/CRA)**

## Business Problem & Goals
In Fintech, a **False Negative** (approving a loan for someone who will default) is much costlier than a **False Positive**. 
* **Goal:** Increase the **Recall** for high-risk applicants.
* **Impact:** The optimized model successfully identified high-risk profiles even in imbalanced datasets, providing a color-coded risk verdict for non-technical stakeholders.

## Tech Stack
* **Modeling:** Python, XGBoost, Scikit-Learn
* **Imbalance Handling:** Imbalanced-Learn (SMOTE)
* **Optimization:** RandomizedSearchCV (Hyperparameter Tuning)
* **Visualization:** Plotly (Interative Gauge Charts)
* **Deployment:** Streamlit & Hugging Face Spaces

## The Data Science Pipeline
1. **Advanced Data Cleaning:** Identified and resolved "impossible" outliers (e.g., age 144 years, employment 123 years) and handled missing interest rate values using median imputation.
2. **Feature Engineering:** Implemented One-Hot Encoding and addressed multicollinearity by dropping highly correlated features (`person_age` vs `cred_hist_length`).
3. **Imbalance Handling:** Applied **SMOTE** to balance the training set, ensuring the model is sensitive to both 'Default' and 'Non-default' classes.
4. **Hyperparameter Tuning:** Optimized XGBoost using `RandomizedSearchCV`, focusing on improving the **F1-Score** and **Recall** for risk detection.

## Key Insights
* **Primary Determinants:** The top 3 predictors of risk found were: **Loan Intent (Medical)**, **Annual Income**, and **Loan Grade**.
* **Model Performance:** Achieved a significant boost in identifying high-risk applicants, reducing potential credit losses.

## Repository Structure
* `Fintech_Loan_Risk_Engine.ipynb`: Data exploration, cleaning, and model training.
* `app.py`: Streamlit application script.
* `credit_risk_model.pkl`: Exported XGBoost model.
* `requirements.txt`: Environment dependencies.

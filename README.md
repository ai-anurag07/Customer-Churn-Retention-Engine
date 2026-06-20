# 📊 Customer Churn Prediction & Retention Strategy Engine

## 📌 Project Overview
This project is an end-to-end Machine Learning and Business Intelligence solution designed to predict customer churn, identify high-risk accounts, and automatically recommend business retention strategies. It transitions standard predictive modeling into an actionable, business-facing Streamlit dashboard.

## 🚀 Core Features
* **Phase 1: Predictive Modeling:** Built an `XGBoost` classifier to predict churn probability, achieving an **0.83 ROC-AUC**.
* **Phase 2: Customer Segmentation:** Applied `K-Means Clustering` to segment customers based on value, tenure, and risk profiles.
* **Phase 3: Customer Lifetime Value (CLV):** Estimated revenue at risk for every individual customer based on current monthly charges and churn probability.
* **Phase 4: Retention Engine:** Developed algorithmic business rules to assign specific, actionable retention strategies (e.g., "Offer 20% Annual Discount" or "Trigger Personal Support Call") based on the customer's specific risk factors.
* **Phase 5: Interactive UI:** Built a front-end `Streamlit` web application allowing stakeholders to search for Customer IDs and view risk profiles and recommended actions in real-time.

## 🛠️ Tech Stack
* **Languages:** Python
* **Machine Learning:** XGBoost, Scikit-Learn (K-Means, Random Forest)
* **Data Processing:** Pandas, NumPy
* **Frontend / UI:** Streamlit, Plotly
* **Data Source:** Official IBM Telco Customer Churn Dataset

## ⚙️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ai-anurag07/Customer-Churn-Retention-Engine.git
Install dependencies:
pip install -r requirements.txt

Run the Machine Learning backend to train the model and generate the dashboard data:
python model_pipeline.py

Launch the interactive dashboard:
python -m streamlit run app.py

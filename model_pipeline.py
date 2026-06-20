import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    print("1. Downloading and loading data...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    print("2. Preprocessing data...")
    # Fix TotalCharges (contains blank spaces for 0 tenure)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert Churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def feature_engineering(df):
    print("3. Feature Engineering & Encoding...")
    # Create a copy for ML
    df_ml = df.copy()
    
    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = df_ml.select_dtypes(include=['object']).columns
    cat_cols = cat_cols.drop('customerID') # Keep ID intact
    
    for col in cat_cols:
        df_ml[col] = le.fit_transform(df_ml[col])
        
    return df_ml

def phase_1_prediction(df_ml):
    print("4. Training Churn Models (Phase 1)...")
    X = df_ml.drop(columns=['customerID', 'Churn'])
    y = df_ml['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost as our primary model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    preds = xgb_model.predict(X_test)
    probs = xgb_model.predict_proba(X_test)[:, 1]
    
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print("Classification Report:\n", classification_report(y_test, preds))
    
    # Save the model and the feature names
    joblib.dump(xgb_model, 'xgb_churn_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_features.pkl')
    
    # Get predictions for the ENTIRE dataset for our dashboard
    df_ml['Churn_Probability'] = xgb_model.predict_proba(X)[:, 1]
    return df_ml

def phase_3_segmentation(df_ml):
    print("5. Customer Segmentation via K-Means (Phase 3)...")
    # Clustering based on value and tenure
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_ml[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_ml['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Map clusters to readable names
    cluster_map = {
        0: 'Low Value / Low Tenure',
        1: 'High Value / High Tenure',
        2: 'High Value / Low Tenure (High Risk)'
    }
    df_ml['Segment_Name'] = df_ml['Cluster'].map(cluster_map)
    return df_ml

def phase_4_and_5_business_logic(df, df_ml):
    print("6. Calculating CLV and Applying Business Rules (Phases 4 & 5)...")
    # Bring predictions back to the original readable dataframe
    df['Churn_Probability'] = df_ml['Churn_Probability']
    df['Segment_Name'] = df_ml['Segment_Name']
    
    # Phase 4: CLV Estimation (Monthly Charges * expected lifespan in months)
    # Expected lifespan = 1 / (monthly churn rate). Using Churn Prob as a proxy.
    # Cap probability at 0.01 to avoid dividing by zero
    capped_prob = df['Churn_Probability'].clip(lower=0.01)
    df['Estimated_CLV'] = df['MonthlyCharges'] * (1 / capped_prob)
    
    # Phase 5: Recommendation Engine Business Rules
    def get_action(row):
        if row['Churn_Probability'] > 0.7 and row['Contract'] == 'Month-to-month':
            return "Offer 20% discount on Annual Upgrade"
        elif row['Churn_Probability'] > 0.8 and row['tenure'] < 6:
            return "High Flight Risk: Trigger personal support call"
        elif row['InternetService'] == 'Fiber optic' and row['Churn_Probability'] > 0.6:
            return "Offer free Tech Support visit / Router check"
        elif row['Churn_Probability'] < 0.2:
            return "Low Risk: Upsell Premium Channels"
        else:
            return "Monitor Account"
            
    df['Retention_Action'] = df.apply(get_action, axis=1)
    return df

def main():
    df = load_and_clean_data()
    df_ml = feature_engineering(df)
    df_ml = phase_1_prediction(df_ml)
    df_ml = phase_3_segmentation(df_ml)
    
    final_df = phase_4_and_5_business_logic(df, df_ml)
    
    # Save the final dataset for the dashboard
    final_df.to_csv('dashboard_data.csv', index=False)
    print("Pipeline Complete! 'dashboard_data.csv' and 'xgb_churn_model.pkl' saved.")

if __name__ == "__main__":
    main()
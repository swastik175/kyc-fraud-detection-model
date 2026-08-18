import numpy as np
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------------
# PHASE 1: CUSTOM DATA TRANSFORMER (FEATURE ENGINEERING)
# -------------------------------------------------------------------------
class FintechFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms raw FinTech KYC fields into numeric risk features
    suitable for Logistic Regression.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Create a deep copy to prevent altering original data frame
        X_out = pd.DataFrame(index=X.index)

        # 1. Process DOB to Age
        current_year = 2026
        X_out['age'] = current_year - pd.to_datetime(X['dob']).dt.year

        # 2. PAN Format Validation (Pattern: 5 Letters, 4 Digits, 1 Letter)
        pan_regex = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        X_out['is_pan_valid'] = X['pan_number'].apply(
            lambda x: 1 if re.match(pan_regex, str(x).upper()) else 0
        )

        # 3. Aadhaar Format Validation (12 Digits)
        aadhaar_regex = r'^[0-9]{12}$'
        X_out['is_aadhaar_valid'] = X['aadhaar_number'].apply(
            lambda x: 1 if re.match(aadhaar_regex, str(x)) else 0
        )

        # 4. Email Domain Risk (Flag free/disposable providers vs corporate/reputable ones)
        risky_domains = ['mailinator.com', 'trashmail.com', 'tempmail.com', 'test.com']
        X_out['is_risky_email'] = X['email'].apply(
            lambda x: 1 if str(x).split('@')[-1].lower() in risky_domains else 0
        )

        # 5. Shop Name Quality (Flagging low-effort names like 'xyz', 'test', 'asdf')
        suspicious_words = ['test', 'asdf', 'xyz', 'none', 'na', 'shop']
        X_out['is_suspicious_shop_name'] = X['shop_name'].apply(
            lambda x: 1 if str(x).lower().strip() in suspicious_words or len(str(x)) < 4 else 0
        )

        # 6. Location Distance Anomaly (Approximated Calculation)
        # In production, use Haversine formula mapping Lat/Long to actual Pincode centroid.
        # Here we simulate if GPS coordinates deviate from regional expectations.
        X_out['gps_distance_anomaly'] = X['location_deviation_score']

        # 7. Convert Gender to Binary Numeric (0 = Male, 1 = Female, 2 = Other)
        X_out['gender_encoded'] = X['gender'].map({'Male': 0, 'Female': 1, 'Other': 2}).fillna(0)

        return X_out

# -------------------------------------------------------------------------
# PHASE 2: DATA SIMULATION & GENERATION
# -------------------------------------------------------------------------
def generate_indian_kyc_dataset(num_samples=5000):
    np.random.seed(42)

    # Simulating the exact raw payload parameters requested
    data = {
        'aadhaar_number': np.random.choice(['123456789012', '987654321098', 'INVALID123'], size=num_samples, p=[0.45, 0.45, 0.10]),
        'pan_number': np.random.choice(['ABCDE1234F', 'XYZW9876G', 'BADPAN123'], size=num_samples, p=[0.45, 0.45, 0.10]),
        'dob': np.random.choice(['1990-05-12', '1985-11-23', '2010-01-01'], size=num_samples), # 2010 makes them underage in 2026
        'gender': np.random.choice(['Male', 'Female'], size=num_samples),
        'latitude': np.random.uniform(8.4, 37.6, size=num_samples),
        'longitude': np.random.uniform(68.7, 97.2, size=num_samples),
        'shop_name': np.random.choice(['Kiran Kirana Store', 'Sharma Electronics', 'test', 'xyz'], size=num_samples, p=[0.4, 0.4, 0.1, 0.1]),
        'city': np.random.choice(['Mumbai', 'Delhi', 'Bengaluru'], size=num_samples),
        'state': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka'], size=num_samples),
        'pincode': np.random.choice(['400001', '110001', '560001'], size=num_samples),
        'email': np.random.choice(['user@gmail.com', 'merchant@yahoo.com', 'fraud@mailinator.com'], size=num_samples, p=[0.45, 0.45, 0.10]),
        'mobile_number': np.random.choice(['9876543210', '8765432109', '12345'], size=num_samples, p=[0.45, 0.45, 0.10]),
        # System tracking value comparing GPS against regional Pin Code matrix
        'location_deviation_score': np.random.uniform(0, 1, size=num_samples)
    }

    df = pd.DataFrame(data)

    # Establish dynamic rules for Ground Truth Labels
    # Fraud condition: Bad document formatting, fake shop names, or massive location deviation
    is_fraud = (
        (df['pan_number'] == 'BADPAN123').astype(int) * 0.4 +
        (df['aadhaar_number'] == 'INVALID123').astype(int) * 0.4 +
        (df['email'] == 'fraud@mailinator.com').astype(int) * 0.3 +
        (df['location_deviation_score'] > 0.75).astype(int) * 0.3
    )

    df['is_fraud'] = (is_fraud > 0.5).astype(int)
    return df

# -------------------------------------------------------------------------
# PHASE 3: PIPELINE EXECUTION & TRAINING
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("⏳ Step 1: Generating Raw Indian FinTech KYC Dataset...")
    raw_df = generate_indian_kyc_dataset(5000)

    X = raw_df.drop(columns=['is_fraud'])
    y = raw_df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n⏳ Step 2: building end-to-end ML Pipeline (Transformation -> Scaling -> Optimization)...")
    # Structuring execution path
    fintech_risk_pipeline = Pipeline([
        ('feature_engineering', FintechFeatureExtractor()), # Converts text strings to numbers
        ('scaler', StandardScaler()),                       # Normalizes numerical outputs
        ('classifier', LogisticRegression(
            solver='newton-cg',                             # Newton's Optimization Method
            class_weight='balanced',                        # Adjusts for low fraud base rates
            random_state=42
        ))
    ])

    print("⏳ Step 3: Training Model via Newton-CG Solver...")
    fintech_risk_pipeline.fit(X_train, y_train)

    print("\n⏳ Step 4: System Performance Evaluation...")
    predictions = fintech_risk_pipeline.predict(X_test)
    probabilities = fintech_risk_pipeline.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification Matrix Metrics:")
    print(classification_report(y_test, predictions))

    # -------------------------------------------------------------------------
    # PHASE 4: INTERCEPTING A LIVE KYC WEBHOOK CORRUPTED PAYLOAD
    # -------------------------------------------------------------------------
    print("\n🚀 Step 5: Testing Production Webhook Interception Pipeline...")

    # High-Risk payload incoming from frontend application portal
    suspicious_payload = pd.DataFrame([{
        'aadhaar_number': '123456',               # Invalid length
        'pan_number': 'FGGHH',               # Synthetically malformed PAN
        'dob': '2012-05-15',                     # Underage user profile (14 years old in 2026)
        'gender': 'Male',
        'latitude': 19.0760,
        'longitude': 72.8777,
        'shop_name': 'xyz',                      # Nonsense placeholder shop name
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'pincode': '400001',
        'email': 'attacker@mailinator.com',      # Flagged disposable domain match
        'mobile_number': '9999999999',
        'location_deviation_score': 0.12         # Device IP/GPS deviates 800km away from Mumbai Pincode
    }])

    # Evaluate live transaction payload directly through saved pipeline setup
    risk_probability = fintech_risk_pipeline.predict_proba(suspicious_payload)[0][1]
    print(f"🔴 Live Risk Firewall Analysis: Fraud Probability Score is {risk_probability * 100:.2f}%")

    if risk_probability > 0.70:
        print("🛑 ACTION REJECTED: Automated KYC onboarding intercepted. Terminating bank routing protocol.")

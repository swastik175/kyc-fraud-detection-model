"""
Unit tests for the FinTech KYC fraud detection pipeline.

NOTE: This test file assumes the pipeline script is saved as `model.py`
in the repo root. If your file has a different name (e.g. `train.py`,
`fraud_pipeline.py`), update the import line below to match:

    from model import FintechFeatureExtractor, generate_indian_kyc_dataset

The script's training/demo code lives inside `if __name__ == "__main__":`,
so importing it here does NOT re-run training, printing, or the webhook
simulation — only the class and function definitions are loaded.
"""

import pandas as pd
import pytest

from model import FintechFeatureExtractor, generate_indian_kyc_dataset


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_row():
    """A single, well-formed 'clean' KYC payload."""
    return pd.DataFrame([{
        'aadhaar_number': '123456789012',
        'pan_number': 'ABCDE1234F',
        'dob': '1990-05-12',
        'gender': 'Male',
        'latitude': 19.0760,
        'longitude': 72.8777,
        'shop_name': 'Kiran Kirana Store',
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'pincode': '400001',
        'email': 'user@gmail.com',
        'mobile_number': '9876543210',
        'location_deviation_score': 0.10,
    }])


@pytest.fixture
def suspicious_row():
    """A high-risk payload matching every fraud signal at once."""
    return pd.DataFrame([{
        'aadhaar_number': '123456',            # invalid length
        'pan_number': 'FGGHH',                 # malformed PAN
        'dob': '2012-05-15',                   # underage in 2026
        'gender': 'Male',
        'latitude': 19.0760,
        'longitude': 72.8777,
        'shop_name': 'xyz',                    # suspicious placeholder name
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'pincode': '400001',
        'email': 'attacker@mailinator.com',    # risky domain
        'mobile_number': '9999999999',
        'location_deviation_score': 0.12,
    }])


# -------------------------------------------------------------------------
# FintechFeatureExtractor tests
# -------------------------------------------------------------------------

class TestFintechFeatureExtractor:

    def test_output_columns(self, sample_row):
        out = FintechFeatureExtractor().transform(sample_row)
        expected_cols = {
            'age', 'is_pan_valid', 'is_aadhaar_valid', 'is_risky_email',
            'is_suspicious_shop_name', 'gps_distance_anomaly', 'gender_encoded',
        }
        assert expected_cols.issubset(set(out.columns))

    def test_age_calculation(self, sample_row):
        out = FintechFeatureExtractor().transform(sample_row)
        # dob is 1990-05-12; script hardcodes current_year = 2026
        assert out['age'].iloc[0] == 2026 - 1990

    def test_valid_pan_and_aadhaar_flagged_correctly(self, sample_row):
        out = FintechFeatureExtractor().transform(sample_row)
        assert out['is_pan_valid'].iloc[0] == 1
        assert out['is_aadhaar_valid'].iloc[0] == 1

    def test_invalid_pan_and_aadhaar_flagged_correctly(self, suspicious_row):
        out = FintechFeatureExtractor().transform(suspicious_row)
        assert out['is_pan_valid'].iloc[0] == 0
        assert out['is_aadhaar_valid'].iloc[0] == 0

    def test_risky_email_domain_detected(self, suspicious_row):
        out = FintechFeatureExtractor().transform(suspicious_row)
        assert out['is_risky_email'].iloc[0] == 1

    def test_clean_email_not_flagged(self, sample_row):
        out = FintechFeatureExtractor().transform(sample_row)
        assert out['is_risky_email'].iloc[0] == 0

    def test_suspicious_shop_name_detected(self, suspicious_row):
        out = FintechFeatureExtractor().transform(suspicious_row)
        assert out['is_suspicious_shop_name'].iloc[0] == 1

    def test_legit_shop_name_not_flagged(self, sample_row):
        out = FintechFeatureExtractor().transform(sample_row)
        assert out['is_suspicious_shop_name'].iloc[0] == 0

    def test_short_shop_name_flagged_regardless_of_content(self):
        row = pd.DataFrame([{
            'aadhaar_number': '123456789012',
            'pan_number': 'ABCDE1234F',
            'dob': '1990-05-12',
            'gender': 'Female',
            'latitude': 19.0,
            'longitude': 72.0,
            'shop_name': 'ab',  # < 4 chars, but not in suspicious_words list
            'city': 'Mumbai',
            'state': 'Maharashtra',
            'pincode': '400001',
            'email': 'user@gmail.com',
            'mobile_number': '9876543210',
            'location_deviation_score': 0.10,
        }])
        out = FintechFeatureExtractor().transform(row)
        assert out['is_suspicious_shop_name'].iloc[0] == 1

    def test_gender_encoding(self):
        row = pd.DataFrame([
            {'aadhaar_number': '123456789012', 'pan_number': 'ABCDE1234F',
             'dob': '1990-05-12', 'gender': 'Male', 'latitude': 19.0,
             'longitude': 72.0, 'shop_name': 'Sharma Electronics', 'city': 'Mumbai',
             'state': 'Maharashtra', 'pincode': '400001', 'email': 'user@gmail.com',
             'mobile_number': '9876543210', 'location_deviation_score': 0.1},
            {'aadhaar_number': '123456789012', 'pan_number': 'ABCDE1234F',
             'dob': '1990-05-12', 'gender': 'Female', 'latitude': 19.0,
             'longitude': 72.0, 'shop_name': 'Sharma Electronics', 'city': 'Mumbai',
             'state': 'Maharashtra', 'pincode': '400001', 'email': 'user@gmail.com',
             'mobile_number': '9876543210', 'location_deviation_score': 0.1},
        ])
        out = FintechFeatureExtractor().transform(row)
        assert list(out['gender_encoded']) == [0, 1]

    def test_gps_distance_anomaly_passthrough(self, sample_row):
        out = FintechFeatureExtractor().transform(sample_row)
        assert out['gps_distance_anomaly'].iloc[0] == pytest.approx(0.10)

    def test_transform_does_not_mutate_input(self, sample_row):
        original = sample_row.copy(deep=True)
        FintechFeatureExtractor().transform(sample_row)
        pd.testing.assert_frame_equal(sample_row, original)

    def test_fit_returns_self(self, sample_row):
        extractor = FintechFeatureExtractor()
        assert extractor.fit(sample_row) is extractor


# -------------------------------------------------------------------------
# generate_indian_kyc_dataset tests
# -------------------------------------------------------------------------

class TestGenerateIndianKycDataset:

    def test_shape(self):
        df = generate_indian_kyc_dataset(num_samples=200)
        assert len(df) == 200

    def test_required_columns_present(self):
        df = generate_indian_kyc_dataset(num_samples=50)
        expected = {
            'aadhaar_number', 'pan_number', 'dob', 'gender', 'latitude',
            'longitude', 'shop_name', 'city', 'state', 'pincode', 'email',
            'mobile_number', 'location_deviation_score', 'is_fraud',
        }
        assert expected.issubset(set(df.columns))

    def test_is_fraud_is_binary(self):
        df = generate_indian_kyc_dataset(num_samples=500)
        assert set(df['is_fraud'].unique()).issubset({0, 1})

    def test_reproducible_with_fixed_seed(self):
        # np.random.seed(42) is set inside the function, so two calls
        # with the same sample size should be identical.
        df1 = generate_indian_kyc_dataset(num_samples=100)
        df2 = generate_indian_kyc_dataset(num_samples=100)
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_null_values(self):
        df = generate_indian_kyc_dataset(num_samples=100)
        assert df.isnull().sum().sum() == 0

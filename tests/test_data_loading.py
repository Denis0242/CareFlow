import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.careflow_utils import generate_data, load_data


def test_generate_data_returns_dataframe():
    df = generate_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_generate_data_has_expected_columns():
    df = generate_data()

    expected_columns = [
        "user_id",
        "role",
        "department",
        "plan",
        "tenure_months",
        "logins_last_30d",
        "session_length_min",
        "patients_documented",
        "features_adopted",
        "ttfv_days",
        "support_tickets",
        "nps_score",
        "adopted_patient_timeline",
        "adopted_e_prescribing",
        "adopted_care_gap_alerts",
        "adopted_secure_messaging",
        "adopted_lab_viewer",
        "adopted_referral_tracker",
        "adopted_voice_notes",
        "adopted_analytics_dashboard",
        "depth_patient_timeline",
        "depth_e_prescribing",
        "depth_care_gap_alerts",
        "depth_secure_messaging",
        "depth_voice_notes",
        "depth_analytics_dashboard",
        "cohort_month",
        "power_user",
    ]

    for col in expected_columns:
        assert col in df.columns


def test_power_user_is_binary():
    df = generate_data()
    assert set(df["power_user"].unique()).issubset({0, 1})


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
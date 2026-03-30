from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "synthetic_careflow_data.csv"

FEATURES_LIST = [
    "patient_timeline",
    "e_prescribing",
    "care_gap_alerts",
    "secure_messaging",
    "lab_viewer",
    "referral_tracker",
    "voice_notes",
    "analytics_dashboard",
]

FEATURE_LABELS = {
    "patient_timeline": "Patient Timeline",
    "e_prescribing": "e-Prescribing",
    "care_gap_alerts": "Care Gap Alerts",
    "secure_messaging": "Secure Messaging",
    "lab_viewer": "Lab Result Viewer",
    "referral_tracker": "Referral Tracker",
    "voice_notes": "Voice Notes",
    "analytics_dashboard": "Analytics Dashboard",
}

MODEL_FEATURES = [
    "tenure_months",
    "logins_last_30d",
    "session_length_min",
    "patients_documented",
    "features_adopted",
    "ttfv_days",
    "support_tickets",
    "nps_score",
    "adopted_e_prescribing",
    "adopted_care_gap_alerts",
    "adopted_voice_notes",
    "adopted_analytics_dashboard",
    "depth_voice_notes",
    "depth_analytics_dashboard",
]

MODEL_FEATURE_DESC = {
    "tenure_months": "Months since account activation",
    "logins_last_30d": "Login events in last 30 days",
    "session_length_min": "Average session length (minutes)",
    "patients_documented": "Patients documented in last 30 days",
    "features_adopted": "Count of distinct features ever used",
    "ttfv_days": "Days to first meaningful action",
    "support_tickets": "Support tickets raised",
    "nps_score": "Net Promoter Score (-100 to 100)",
    "adopted_e_prescribing": "Has used e-Prescribing",
    "adopted_care_gap_alerts": "Has used Care Gap Alerts",
    "adopted_voice_notes": "Has used Voice Notes",
    "adopted_analytics_dashboard": "Has used Analytics Dashboard",
    "depth_voice_notes": "Voice Notes usage depth",
    "depth_analytics_dashboard": "Analytics Dashboard usage depth",
}

ROLE_LIST = [
    "Physician",
    "Nurse Practitioner",
    "Care Coordinator",
    "Registered Nurse",
]

DEPT_LIST = [
    "Primary Care",
    "Cardiology",
    "Oncology",
    "Emergency",
    "Pediatrics",
]

PLAN_LIST = [
    "Starter",
    "Professional",
    "Enterprise",
]


def normalize_shap_values(raw_shap_values, n_features: int) -> np.ndarray:
    arr = None

    if isinstance(raw_shap_values, list):
        candidates = [np.array(x) for x in raw_shap_values]
        for candidate in candidates:
            if candidate.ndim == 2 and candidate.shape[1] == n_features:
                arr = candidate
                break
            if candidate.ndim == 3:
                squeezed = np.squeeze(candidate)
                if squeezed.ndim == 2 and squeezed.shape[1] == n_features:
                    arr = squeezed
                    break
        if arr is None:
            arr = np.array(candidates[-1])
    elif hasattr(raw_shap_values, "values"):
        arr = np.array(raw_shap_values.values)
    else:
        arr = np.array(raw_shap_values)

    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            arr = arr[:, :, -1]
        elif arr.shape[2] == n_features:
            arr = arr[:, -1, :]
        else:
            arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim == 1:
        if arr.size == n_features:
            arr = arr.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected 1D SHAP shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"SHAP values could not be normalized to 2D. Got shape: {arr.shape}")

    if arr.shape[1] != n_features and arr.shape[0] == n_features:
        arr = arr.T

    if arr.shape[1] != n_features:
        raise ValueError(
            f"SHAP feature mismatch after normalization. "
            f"Expected {n_features} features, got shape {arr.shape}."
        )

    return arr


def generate_data(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    roles = rng.choice(ROLE_LIST, n, p=[0.35, 0.25, 0.20, 0.20])
    departments = rng.choice(DEPT_LIST, n)
    plans = rng.choice(PLAN_LIST, n, p=[0.40, 0.40, 0.20])

    plan_idx = np.where(plans == "Starter", 0, np.where(plans == "Professional", 1, 2))

    tenure_months = (rng.exponential(10, n) + 1).clip(1, 36).astype(int)
    logins_last_30d = np.clip(rng.normal(12 + plan_idx * 3, 5, n), 0, 60).astype(int)
    session_length_min = np.clip(rng.normal(14 + plan_idx * 2, 6, n), 1, 90).round(1)
    patients_documented = np.clip(rng.normal(35 + plan_idx * 10, 20, n), 0, 200).astype(int)

    adopted_patient_timeline = (rng.random(n) < 0.90).astype(int)
    adopted_e_prescribing = (rng.random(n) < 0.65 + plan_idx * 0.05).astype(int)
    adopted_care_gap_alerts = (rng.random(n) < 0.50 + plan_idx * 0.08).astype(int)
    adopted_secure_messaging = (rng.random(n) < 0.72).astype(int)
    adopted_lab_viewer = (rng.random(n) < 0.60 + plan_idx * 0.06).astype(int)
    adopted_referral_tracker = (rng.random(n) < 0.40 + plan_idx * 0.07).astype(int)
    adopted_voice_notes = (rng.random(n) < 0.28 + plan_idx * 0.10).astype(int)
    adopted_analytics_dashboard = (rng.random(n) < 0.22 + plan_idx * 0.15).astype(int)

    features_adopted = (
        adopted_patient_timeline
        + adopted_e_prescribing
        + adopted_care_gap_alerts
        + adopted_secure_messaging
        + adopted_lab_viewer
        + adopted_referral_tracker
        + adopted_voice_notes
        + adopted_analytics_dashboard
    )

    def depth(adopted: np.ndarray, base: float, boost: np.ndarray) -> np.ndarray:
        raw = np.clip(rng.normal(base + boost, 2.5, n), 0, 10)
        return (raw * adopted).round(1)

    depth_patient_timeline = depth(adopted_patient_timeline, 7.0, plan_idx * 0.3)
    depth_e_prescribing = depth(adopted_e_prescribing, 6.0, plan_idx * 0.4)
    depth_care_gap_alerts = depth(adopted_care_gap_alerts, 5.5, plan_idx * 0.5)
    depth_secure_messaging = depth(adopted_secure_messaging, 6.5, plan_idx * 0.2)
    depth_voice_notes = depth(adopted_voice_notes, 4.5, plan_idx * 0.8)
    depth_analytics_dashboard = depth(adopted_analytics_dashboard, 4.0, plan_idx * 1.0)

    ttfv_days = np.clip(rng.exponential(4, n) + 1, 1, 30).astype(int)
    support_tickets = np.clip(rng.poisson(0.8, n), 0, 8)
    nps_score = np.clip(rng.normal(42 + plan_idx * 8, 22, n), -100, 100).astype(int)

    cohort_month = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 18, n) * 30,
        unit="D",
    )

    score = (
        0.25 * (logins_last_30d / 30)
        + 0.20 * (features_adopted / 8)
        + 0.15 * (session_length_min / 60)
        + 0.10 * (depth_voice_notes / 10)
        + 0.10 * (depth_analytics_dashboard / 10)
        + 0.08 * (1 - ttfv_days / 30)
        + 0.07 * ((nps_score / 100 + 1) / 2)
        - 0.05 * (support_tickets / 8)
        + rng.normal(0, 0.08, n)
    )

    power_user = (score > np.percentile(score, 45)).astype(int)

    return pd.DataFrame(
        {
            "user_id": [f"CF{i:05d}" for i in range(n)],
            "role": roles,
            "department": departments,
            "plan": plans,
            "tenure_months": tenure_months,
            "logins_last_30d": logins_last_30d,
            "session_length_min": session_length_min,
            "patients_documented": patients_documented,
            "features_adopted": features_adopted,
            "ttfv_days": ttfv_days,
            "support_tickets": support_tickets,
            "nps_score": nps_score,
            "adopted_patient_timeline": adopted_patient_timeline,
            "adopted_e_prescribing": adopted_e_prescribing,
            "adopted_care_gap_alerts": adopted_care_gap_alerts,
            "adopted_secure_messaging": adopted_secure_messaging,
            "adopted_lab_viewer": adopted_lab_viewer,
            "adopted_referral_tracker": adopted_referral_tracker,
            "adopted_voice_notes": adopted_voice_notes,
            "adopted_analytics_dashboard": adopted_analytics_dashboard,
            "depth_patient_timeline": depth_patient_timeline,
            "depth_e_prescribing": depth_e_prescribing,
            "depth_care_gap_alerts": depth_care_gap_alerts,
            "depth_secure_messaging": depth_secure_messaging,
            "depth_voice_notes": depth_voice_notes,
            "depth_analytics_dashboard": depth_analytics_dashboard,
            "cohort_month": cohort_month,
            "power_user": power_user,
        }
    )


def save_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = path or DEFAULT_DATA_PATH
    df.to_csv(output_path, index=False)
    return output_path


def load_data(path: Path | None = None) -> pd.DataFrame:
    data_path = path or DEFAULT_DATA_PATH
    if data_path.exists():
        return pd.read_csv(data_path, parse_dates=["cohort_month"])
    df = generate_data()
    save_data(df, data_path)
    return df


def train_models(df: pd.DataFrame) -> dict:
    X = df[MODEL_FEATURES].copy()
    y = df["power_user"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.07,
        max_depth=3,
        random_state=42,
    )
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    lr.fit(X_tr_scaled, y_tr)

    def metric_row(name: str, model, x_eval, y_eval) -> dict:
        preds = model.predict(x_eval)
        probs = model.predict_proba(x_eval)[:, 1]
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_eval, preds),
            "Precision": precision_score(y_eval, preds, zero_division=0),
            "Recall": recall_score(y_eval, preds, zero_division=0),
            "F1": f1_score(y_eval, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_eval, probs),
        }

    results = (
        pd.DataFrame(
            [
                metric_row("Random Forest", rf, X_te, y_te),
                metric_row("Gradient Boosting", gb, X_te, y_te),
                metric_row("Logistic Regression", lr, X_te_scaled, y_te),
            ]
        )
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

    rf_fi = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    gb_fi = (
        pd.DataFrame({"feature": X.columns, "importance": gb.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    cm_rf = confusion_matrix(y_te, rf.predict(X_te))
    cm_gb = confusion_matrix(y_te, gb.predict(X_te))
    cm_lr = confusion_matrix(y_te, lr.predict(X_te_scaled))

    shap_sample = X_tr.sample(min(120, len(X_tr)), random_state=42)
    explainer = None
    shap_arr = None
    shap_error = None

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(rf)
            raw_shap = explainer.shap_values(shap_sample)
            shap_arr = normalize_shap_values(raw_shap, n_features=shap_sample.shape[1])
        except Exception as exc:
            shap_arr = None
            shap_error = str(exc)

    best_name = results.iloc[0]["Model"]
    if best_name == "Random Forest":
        best_model = rf
    elif best_name == "Gradient Boosting":
        best_model = gb
    else:
        best_model = lr

    best_scale = best_name == "Logistic Regression"

    return {
        "X": X,
        "y": y,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_te": y_te,
        "rf": rf,
        "gb": gb,
        "lr": lr,
        "scaler": scaler,
        "results": results,
        "rf_fi": rf_fi,
        "gb_fi": gb_fi,
        "cm_rf": cm_rf,
        "cm_gb": cm_gb,
        "cm_lr": cm_lr,
        "shap_sample": shap_sample,
        "explainer": explainer,
        "shap_arr": shap_arr,
        "shap_error": shap_error,
        "best_name": best_name,
        "best_model": best_model,
        "best_scale": best_scale,
    }
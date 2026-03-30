import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.careflow_utils import generate_data, train_models


def test_train_models_returns_expected_keys():
    df = generate_data()
    artifacts = train_models(df)

    expected_keys = [
        "X",
        "y",
        "X_tr",
        "X_te",
        "y_te",
        "rf",
        "gb",
        "lr",
        "scaler",
        "results",
        "rf_fi",
        "gb_fi",
        "cm_rf",
        "cm_gb",
        "cm_lr",
        "shap_sample",
        "explainer",
        "shap_arr",
        "shap_error",
        "best_name",
        "best_model",
        "best_scale",
    ]

    for key in expected_keys:
        assert key in artifacts


def test_results_is_dataframe_and_not_empty():
    df = generate_data()
    artifacts = train_models(df)

    results = artifacts["results"]
    assert isinstance(results, pd.DataFrame)
    assert not results.empty


def test_results_contains_required_metrics():
    df = generate_data()
    artifacts = train_models(df)

    results = artifacts["results"]
    required_columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    for col in required_columns:
        assert col in results.columns


def test_metric_values_are_in_valid_range():
    df = generate_data()
    artifacts = train_models(df)

    results = artifacts["results"]

    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for col in metric_columns:
        assert results[col].between(0, 1).all()
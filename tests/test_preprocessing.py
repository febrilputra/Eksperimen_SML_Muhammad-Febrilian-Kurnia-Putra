import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("preprocess", ROOT / "preprocessing" / "automate_Muhammad-Febrilian-Kurnia-Putra.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_customer_split_and_training_only_statistics():
    raw = MODULE.load_raw(ROOT / "telco_churn_raw" / "Telco-Customer-Churn.csv")
    splits = MODULE.split_raw(raw)
    ids = [set(frame.customerID) for frame in splits.values()]
    assert sum(map(len, ids)) == len(raw) == 7043
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    clean_train = MODULE.clean_features(splits["train"])
    transformer = MODULE.make_preprocessor().fit(clean_train)
    observed = transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_
    np.testing.assert_allclose(observed, clean_train[MODULE.NUMERIC].median().to_numpy())
    assert "customerID" not in transformer.get_feature_names_out()
    assert "Churn" not in transformer.get_feature_names_out()


def test_unseen_category_and_blank_charges_are_valid():
    raw = MODULE.load_raw(ROOT / "telco_churn_raw" / "Telco-Customer-Churn.csv")
    splits = MODULE.split_raw(raw)
    transformer = MODULE.make_preprocessor().fit(MODULE.clean_features(splits["train"]))
    unseen = splits["test"].head(1).copy()
    unseen.loc[:, "PaymentMethod"] = "unseen-method"
    unseen.loc[:, "tenure"] = 0
    unseen.loc[:, "TotalCharges"] = " "
    clean = MODULE.clean_features(unseen)
    assert clean.TotalCharges.iloc[0] == 0
    assert np.isfinite(transformer.transform(clean)).all()


def test_invalid_numeric_is_rejected():
    raw = MODULE.load_raw(ROOT / "telco_churn_raw" / "Telco-Customer-Churn.csv").head(1)
    raw.loc[:, "TotalCharges"] = "not-a-number"
    with pytest.raises(ValueError, match="Non-numeric"):
        MODULE.clean_features(raw)

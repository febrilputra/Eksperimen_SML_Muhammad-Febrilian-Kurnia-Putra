"""Reproduce the Telco notebook preprocessing without interactive notebook state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv"
SOURCE_COMMIT = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
RAW_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [column for column in RAW_FEATURES if column not in NUMERIC]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"customerID": str, "TotalCharges": str})
    expected = ["customerID", *RAW_FEATURES, "Churn"]
    if list(frame.columns) != expected:
        raise ValueError("Raw data schema does not match the original IBM Telco CSV.")
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].str.strip()
    if frame.customerID.isna().any() or frame.customerID.eq("").any():
        raise ValueError("customerID must be nonempty.")
    if frame.customerID.duplicated().any():
        raise ValueError("Repeated customerID requires an explicit grouping/duplicate policy.")
    if set(frame.Churn.unique()) != {"No", "Yes"}:
        raise ValueError("Churn must contain both No and Yes and no unknown labels.")
    return frame


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.loc[:, RAW_FEATURES].copy()
    for column in CATEGORICAL:
        features[column] = features[column].astype("string").str.strip()
        features[column] = features[column].replace("", pd.NA).astype(object)
        features[column] = features[column].where(pd.notna(features[column]), np.nan)
    for column in NUMERIC:
        before = features[column]
        parsed = pd.to_numeric(before, errors="coerce")
        invalid = parsed.isna() & before.notna() & before.astype(str).str.strip().ne("")
        if invalid.any():
            raise ValueError(f"Non-numeric values in {column}; refuse silent conversion.")
        features[column] = parsed
    zero_tenure = features.TotalCharges.isna() & features.tenure.eq(0)
    features.loc[zero_tenure, "TotalCharges"] = 0.0
    return features


def split_raw(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    train, holdout = train_test_split(frame, test_size=0.30, random_state=seed,
                                     stratify=frame.Churn)
    val, test = train_test_split(holdout, test_size=0.50, random_state=seed,
                                stratify=holdout.Churn)
    return {"train": train, "val": val, "test": test}


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                      dtype=np.float64)),
        ]), CATEGORICAL),
    ], verbose_feature_names_out=False)


def export_results(raw_path: Path, splits: dict[str, pd.DataFrame],
                   matrices: dict[str, np.ndarray], preprocessor: ColumnTransformer,
                   output_dir: Path, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = preprocessor.get_feature_names_out().tolist()
    if len(set(names)) != len(names):
        raise ValueError("Encoded feature names must be unique.")
    splits_info = {}
    for split, frame in splits.items():
        matrix = pd.DataFrame(matrices[split], columns=names)
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(f"{split} contains non-finite features.")
        target = frame.Churn.map({"No": 0, "Yes": 1}).reset_index(drop=True)
        matrix.to_csv(output_dir / f"X_{split}.csv", index=False, float_format="%.12g")
        target.to_frame("Churn").to_csv(output_dir / f"y_{split}.csv", index=False)
        splits_info[split] = {
            "rows": len(frame), "encoded_columns": len(names),
            "positive_count": int(target.sum()),
            "customer_ids": frame.customerID.tolist(),
        }
    ids = [set(info["customer_ids"]) for info in splits_info.values()]
    if any(ids[left] & ids[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("A customer occurs in more than one split.")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    schema = {
        "raw_features": RAW_FEATURES, "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL, "encoded_features": names,
        "target": "Churn", "classes": {"No": 0, "Yes": 1},
        "totalcharges_rule": "strip; parse numeric; blank with tenure=0 becomes 0; remaining missing uses training median",
        "categorical_rule": "strip; blank becomes missing; impute training mode; unknown categories encode as zero",
        "sklearn_version": sklearn.__version__,
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    example_raw = splits["test"].iloc[:3].drop(columns=["customerID", "Churn"]).copy()
    example_raw.to_json(output_dir / "example_raw.json", orient="records", indent=2)
    encoded_example = {"dataframe_split": {"columns": names, "data": matrices["test"][:3].tolist()}}
    (output_dir / "example_encoded.json").write_text(json.dumps(encoded_example, indent=2) + "\n", encoding="utf-8")
    files = {path.name: sha256(path) for path in sorted(output_dir.iterdir())
             if path.is_file() and path.name != "data_manifest.json"}
    manifest = {
        "dataset": "IBM Telco Customer Churn", "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT, "raw_sha256": sha256(raw_path),
        "raw_rows": sum(len(frame) for frame in splits.values()), "seed": seed,
        "split_method": f"stratified 70/15/15, customerID disjoint, random_state={seed} for both splits",
        "fit_scope": "training split only", "splits": splits_info, "files": files,
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def preprocess(input_path: Path, output_dir: Path, seed: int = 42) -> dict:
    raw = load_raw(input_path)
    splits = split_raw(raw, seed)
    clean = {name: clean_features(frame) for name, frame in splits.items()}
    preprocessor = make_preprocessor()
    matrices = {"train": preprocessor.fit_transform(clean["train"])}
    matrices.update({name: preprocessor.transform(clean[name]) for name in ("val", "test")})
    return export_results(input_path, splits, matrices, preprocessor, output_dir, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPOSITORY / "telco_churn_raw" / "Telco-Customer-Churn.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "telco_churn_preprocessing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = preprocess(args.input.resolve(), args.output.resolve(), args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "raw_sha256": manifest["raw_sha256"],
                      "splits": {key: {k: v for k, v in value.items() if k != "customer_ids"}
                                 for key, value in manifest["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()

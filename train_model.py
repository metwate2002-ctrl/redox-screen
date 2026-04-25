"""
Model Training — Redox Flow Battery Molecule Screener
Trains a Random Forest classifier to predict redox activity
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "NumRings",
    "NumAromaticRings", "TPSA", "NumRotBonds", "FractionCSP3",
    "NumHeteroatoms", "NumValenceElectrons", "MaxPartialCharge",
    "MinPartialCharge", "NumRadicalElectrons", "RingCount",
    "HeavyAtomCount", "NumAromaticBonds", "Chi0", "Chi1", "Kappa1"
]

def train():
    df = pd.read_csv("redox_dataset.csv")
    X = df[FEATURES].fillna(0)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=2,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

    print("=" * 50)
    print("     MODEL TRAINING COMPLETE")
    print("=" * 50)
    print(f"\n📊 Test Accuracy   : {pipeline.score(X_test, y_test):.2f}")
    print(f"📈 ROC-AUC Score   : {roc_auc_score(y_test, y_prob):.2f}")
    print(f"🔁 CV ROC-AUC (5x) : {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Poor Candidate", "Good Candidate"]))

    # Feature importance
    rf = pipeline.named_steps["model"]
    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    print("\n🔬 Top 5 Important Features:")
    print(importances.nlargest(5).to_string())

    # Save model
    with open("redox_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("\n✅ Model saved as redox_model.pkl")

    # Save feature list
    with open("features.pkl", "wb") as f:
        pickle.dump(FEATURES, f)

    return pipeline

if __name__ == "__main__":
    train()

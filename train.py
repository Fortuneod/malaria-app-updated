"""
train.py — Malaria Severity Prediction Model
"""
import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

print("=" * 60)
print("  MALARIA SEVERITY PREDICTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("Malaria-Data.csv")
print(f"\n[DATA]  Loaded {df.shape[0]} patients, {df.shape[1]} columns")

TARGET   = "severe_maleria"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"[DATA]  Features  : {len(FEATURES)} total")
print(f"[DATA]  Balance   — 0: {(y==0).sum()}  |  1: {(y==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\n[SPLIT] Train: {len(X_train)}  |  Test: {len(X_test)}")

print("\n[MODELS] 5-fold CV, ROC-AUC, SMOTE inside each fold ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "SVM (RBF)":           SVC(probability=True, random_state=42),
}

results = {}
for name, clf in candidates.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)), ("model", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    results[name] = scores.mean()
    print(f"  {name:<28}  AUC = {scores.mean():.4f}  (±{scores.std():.4f})")

best_name = max(results, key=results.get)
best_clf  = candidates[best_name]
print(f"\n[BEST]  '{best_name}' selected  (CV AUC = {results[best_name]:.4f})")

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("model",  best_clf)
])
final_pipeline.fit(X_train, y_train)

print("\n[EVAL]  Test-set performance:")
y_pred  = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm  = confusion_matrix(y_test, y_pred)

print(f"  Accuracy : {acc:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
print(f"                Pred:0   Pred:1")
print(f"  Actual:0   TN={cm[0,0]:<5}  FP={cm[0,1]}")
print(f"  Actual:1   FN={cm[1,0]:<5}  TP={cm[1,1]}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Not Severe','Severe'])}")

model_step = final_pipeline.named_steps["model"]
if hasattr(model_step, "feature_importances_"):
    importances = pd.Series(model_step.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("[IMPORTANCE] Top predictors:")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<20} {bar:<30}  {imp:.4f}")

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model")
os.makedirs(model_path, exist_ok=True)

joblib.dump(final_pipeline, "model/pipeline.joblib")
joblib.dump(FEATURES,       "model/features.joblib")
print("\n[SAVED] model/pipeline.joblib  &  model/features.joblib")
print("  Done. Run  python app.py  to start the API.\n")

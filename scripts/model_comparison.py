# model_comparison.py
import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

DATA_PATH = "../data/preprocessed_data.xlsx"
OUT_DIR = "../results/modeling"
MODEL_DIR = "../models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare(target_col='Target'):
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows.")

    # If no target column exists, create one (choose rule)
    if target_col not in df.columns:
        if 'Time_Diff' in df.columns:
            # create target: failure within 90 days
            df[target_col] = (pd.to_numeric(df['Time_Diff'], errors='coerce') <= 90).fillna(0).astype(int)
            print(f"Created {target_col} from Time_Diff <= 90 days.")
        elif 'Failure rate' in df.columns:
            df[target_col] = (df['Failure rate'] > 0).astype(int)
            print(f"Created {target_col} from Failure rate > 0.")
        else:
            raise ValueError("No suitable column to create target. Add a binary target column named 'Target'.")

    # Select feature columns (drop non-numeric identifiers)
    X = df[['Priority', 'Estimated costs', 'Failure rate', 'MTBF', 'MTTR']].copy()
    # one-hot top categories of FunctLocDescrip.
    top = df['FunctLocDescrip.'].value_counts().nlargest(10).index
    df['FunctLocDescrip._reduced'] = df['FunctLocDescrip.'].where(df['FunctLocDescrip.'].isin(top), 'Other')
    X = pd.concat([X.reset_index(drop=True), pd.get_dummies(df['FunctLocDescrip._reduced'], drop_first=True).reset_index(drop=True)], axis=1).fillna(0)
    y = df[target_col].astype(int)
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    }
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "CV_Mean_Acc": cv.mean(), "CV_Std": cv.std(), "Test_Acc": acc})
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(os.path.join(OUT_DIR, f"{name}_confusion.png"))
        plt.close()
        # save classification report
        with open(os.path.join(OUT_DIR, f"{name}_classification.txt"), 'w') as f:
            f.write(classification_report(y_test, y_pred))
        # ROC (if probability available)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            plt.plot([0,1],[0,1],"--", color="gray")
            plt.title(f"{name} ROC")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.savefig(os.path.join(OUT_DIR, f"{name}_roc.png"))
            plt.close()
        # save model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    print("Saved model comparison table.")
    return X_test, y_test, results_df

def shap_explain(best_model_name, X_test):
    model_path = os.path.join(MODEL_DIR, f"{best_model_name}.pkl")
    model = joblib.load(model_path)
    # use TreeExplainer for tree models else KernelExplainer (simple check)
    if best_model_name in ["RandomForest", "XGBoost", "GradientBoosting"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{best_model_name}_shap_summary.png"))
        plt.close()
        print("Saved SHAP summary.")
    else:
        print("SHAP KernelExplainer for non-tree models is slower; skipping by default.")

if __name__ == "__main__":
    X, y = load_and_prepare()
    X_test, y_test, results_df = train_models(X, y)
    # pick best by CV_Mean_Acc
    best = results_df.sort_values("CV_Mean_Acc", ascending=False).iloc[0]["Model"]
    print("Best model:", best)
    shap_explain(best, X_test)

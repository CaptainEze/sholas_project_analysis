import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def load_data(file_path='../data/preprocessed_data.xlsx', target_col='Target'):
    """
    Load dataset and split into features (X) and target (y).
    """
    try:
        df = pd.read_excel(file_path)
        
        df.columns = df.columns.str.strip()

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")

        print(f"Loaded {len(df)} rows from {file_path}")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_and_compare_models(X, y, output_dir='../results/modeling'):
    """
    Train multiple models, compare performance, and generate SHAP explainability for the best one.
    Saves confusion matrices, classification reports, and a comparison table.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results.append({
            'Model': name,
            'CV_Mean_Accuracy': np.mean(cv_scores),
            'CV_Std': np.std(cv_scores),
            'Test_Accuracy': acc
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'))
        plt.close()

        # Classification Report
        with open(os.path.join(output_dir, f'{name}_classification_report.txt'), 'w') as f:
            f.write(classification_report(y_test, y_pred))

    # Save comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print("Model comparison table saved.")

    # Select best model by CV mean accuracy
    best_model_name = results_df.sort_values(by='CV_Mean_Accuracy', ascending=False).iloc[0]['Model']
    print(f"Best model: {best_model_name}")

    # Train best model for SHAP explainability
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_SHAP_summary.png'))
    plt.close()

    return results_df, best_model_name


X, y = load_data(file_path="../../data/preprocessed_data.xlsx", target_col="Target")

# Run model comparison
if X is not None and y is not None:
    results_df, best_model = train_and_compare_models(X, y, output_dir="../../results/modeling")
    print(results_df)
    print(f"Best model: {best_model}")
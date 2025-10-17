import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import os

def load_data(file_path='../data/preprocessed_data.xlsx'):
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        print(f"Loaded data with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data(df):
    # Create target variable: 1 if failure within 90 days, 0 otherwise
    df['Time_Diff'] = pd.to_numeric(df['Time_Diff'], errors='coerce')
    df['Target'] = (df['Time_Diff'] <= 90).astype(int).where(df['Time_Diff'].notna(), 0)

    # Features
    categorical_features = ['FunctLocDescrip.']
    numeric_features = ['Priority', 'Estimated costs', 'Failure rate', 'MTBF', 'MTTR']
    X_numeric = df[numeric_features].fillna(0)
    
    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_categorical = encoder.fit_transform(df[categorical_features].fillna('Unknown'))
    feature_names = encoder.get_feature_names_out(categorical_features)
    X_categorical = pd.DataFrame(X_categorical, columns=feature_names)

    # Combine features
    X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    y = df['Target']

    return X, y, encoder

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest with class weight
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model

def save_model(model, encoder, file_path='../models/rf_model_improved.pkl'):
    import joblib
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump({'model': model, 'encoder': encoder}, file_path)
    print(f"Model and encoder saved as {file_path}")

if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    if df is not None:
        X, y, encoder = prepare_data(df)
        if len(X) > 0 and len(y) > 0:
            # Train and save model
            model = train_model(X, y)
            save_model(model, encoder)
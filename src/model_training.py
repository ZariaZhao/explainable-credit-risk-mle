import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

def build_pipeline(categorical_features, numerical_features):
    """
    Build a pipeline containing preprocessing and XGBoost classifier.
    """
    # Numerical feature processing: imputation + standardization
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical feature processing: imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Complete pipeline: preprocessing -> model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    return pipeline

def train_and_save(X_train, y_train, X_test, y_test, model_path='models/xgb_baseline.joblib'):
    """
    Train the model and save it to disk.
    """
    # Identify categorical and numerical columns
    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols]
    
    print("Building Pipeline...")
    pipeline = build_pipeline(categorical_cols, numerical_cols)
    
    print("Training Model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Done.")
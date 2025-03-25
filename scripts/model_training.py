import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


def preprocess(df, drop_cols=['id'], target='class'):
    df = df.dropna(subset=[target])  # Drop NaN in target

    X = df.drop(columns=[target] + drop_cols, errors='ignore')
    y = df[target]

    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return as DataFrame with feature names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    return X_train_df, X_test_df, y_train, y_test


def apply_smote(X_train_df, y_train):
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train_df, y_train)

    if isinstance(X_train_df, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train_df.columns)

    return X_resampled, y_resampled

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=12)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1.5,
        scale_pos_weight=100,  # VERY IMPORTANT for imbalanced data
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_xgboost(X_train, y_train, cv=3, n_iter=30, random_state=42):
    """
    Fine-tune XGBoost using RandomizedSearchCV to maximize AUC.

    Returns:
        - best_model: the tuned XGBoost model
        - best_params: best parameter set
        - best_score: best ROC AUC score from CV
    """
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    return best_model, best_params, best_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_proba)
    return report, auc, y_pred, y_proba
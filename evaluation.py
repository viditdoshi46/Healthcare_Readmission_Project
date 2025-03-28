import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance and print metrics and feature importance.
    """
    y_pred = model.predict(X_test)
    print("Test set evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        important_features = [(model.feature_names_in_[idx], importances[idx]) for idx in indices]
        print("\nTop 10 Feature Importances:")
        for feat, imp in important_features:
            print(f"  {feat}: {imp:.4f}")
    elif hasattr(model, "coef_"):
        coef = model.coef_[0]
        feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else range(len(coef))
        top_coef_idx = np.argsort(np.abs(coef))[::-1][:10]
        print("\nTop 10 Model Coefficients (by absolute value):")
        for idx in top_coef_idx:
            print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
    else:
        print("Model type does not support feature importance extraction.")

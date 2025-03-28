import pandas as pd

def encode_categoricals(X_train, X_test, max_categories=20):
    """
    One-hot encode categorical features with grouping for high cardinality.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    categorical_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        unique_vals = X_tr[col].value_counts().index
        if len(unique_vals) > max_categories:
            top_cats = list(unique_vals[:max_categories-1])
            X_tr[col] = X_tr[col].where(X_tr[col].isin(top_cats), other='Other')
            X_te[col] = X_te[col].where(X_te[col].isin(top_cats), other='Other')
        else:
            unseen = ~X_te[col].isin(unique_vals)
            if unseen.any():
                X_te.loc[unseen, col] = 'Other'
    X_tr_enc = pd.get_dummies(X_tr, drop_first=True)
    X_te_enc = pd.get_dummies(X_te, drop_first=True)
    X_tr_enc, X_te_enc = X_tr_enc.align(X_te_enc, join='outer', axis=1, fill_value=0)
    return X_tr_enc, X_te_enc

def create_features(train_df, test_df):
    """
    Split the target and apply categorical encoding to both training and testing data.
    """
    target_col = 'readmission_label'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values
    X_train_enc, X_test_enc = encode_categoricals(X_train, X_test)
    return X_train_enc, X_test_enc, y_train, y_test

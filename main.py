from data_ingestion import load_data
from preprocessing import clean_data
from eda import perform_eda
from feature_engineering import create_features
from modeling import train_model
from evaluation import evaluate_model

if __name__ == "__main__":
    data_file = "data/diabetic_data.csv"
    use_spark = False
    df_raw = load_data(data_file, use_spark=use_spark)
    df_clean = clean_data(df_raw)
    perform_eda(df_clean, output_dir="eda_outputs")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['readmission_label'])
    X_train_enc, X_test_enc, y_train, y_test = create_features(train_df, test_df)
    model, X_test_final, y_test_final = train_model(X_train_enc, y_train, use_spark=False)
    evaluate_model(model, X_test_enc if X_test_final is None else X_test_final, y_test if y_test_final is None else y_test_final)

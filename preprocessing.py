import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean and preprocess the DataFrame by handling missing values, dropping irrelevant columns,
    and converting the target column to binary.
    """
    data = df.copy()
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].str.strip().replace('?', np.nan)
    cols_to_drop = []
    if 'weight' in data.columns:
        cols_to_drop.append('weight')
    if 'encounter_id' in data.columns:
        cols_to_drop.append('encounter_id')
    if 'patient_nbr' in data.columns:
        cols_to_drop.append('patient_nbr')
    data = data.drop(columns=cols_to_drop, errors='ignore')
    data = data.drop_duplicates()
    for col in data.columns:
        if data[col].dtype in [np.int64, np.float64]:
            if data[col].isna().any():
                data[col].fillna(data[col].median(), inplace=True)
        else:
            if data[col].isna().any():
                data[col].fillna('Unknown', inplace=True)
    if 'readmitted' in data.columns:
        data['readmitted'] = data['readmitted'].apply(lambda x: 1 if str(x).strip() == '<30' else 0)
        data.rename(columns={'readmitted': 'readmission_label'}, inplace=True)
    return data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir="eda_outputs"):
    """
    Perform exploratory data analysis by printing dataset info and saving key plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Dataset shape:", df.shape)
    print("Column data types:\n", df.dtypes)
    print("Preview of data:\n", df.head(5))
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    print("\nSummary statistics for numeric features:")
    print(df[numeric_cols].describe())
    if 'readmission_label' in df.columns:
        class_counts = df['readmission_label'].value_counts()
        print("\nTarget class distribution:")
        print(class_counts.to_dict())
        plt.figure(figsize=(4,4))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title("Readmission (30-day) Class Distribution")
        plt.xlabel("Readmitted (1=yes, 0=no)")
        plt.ylabel("Number of Records")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_distribution.png")
        plt.close()
    if 'age' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x='age', hue='readmission_label', data=df)
        plt.title("Age distribution by Readmission status")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/age_vs_readmit.png")
        plt.close()
    print(f"\nEDA plots saved to folder: {output_dir}")

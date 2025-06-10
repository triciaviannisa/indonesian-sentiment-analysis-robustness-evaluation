import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv("/path/to/Formality-Level Dataset Construction (Responses) - Form Responses 1.csv")

politeness_levels = {
    "high_politeness.csv": "formal",
    "medium_politeness.csv": "semi_formal",
    "low_politeness.csv": "informal"
}

for filename, col_name in politeness_levels.items():
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found!")
        continue
    
    df_subset = df[[col_name, "gold_label"]].dropna().copy()
    df_subset['gold_label'] = df_subset['gold_label'].astype(int)
    df_subset = df_subset.rename(columns={col_name: "sentence"})
    df_subset["sentence"] = df_subset["sentence"].apply(lambda x: " ".join(word_tokenize(str(x).lower())))
    df_subset.to_csv(filename, index=False)
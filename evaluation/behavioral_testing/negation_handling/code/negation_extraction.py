import pandas as pd

def negation_extraction(new_negation, save_path):

    df = pd.read_csv("/path/to/smsa_dataset_changed_label.csv")

    df_filtered = df[df["sentence"].str.contains(r"\btidak\b", case=False, regex=True)]

    df_filtered["sentence"] = df_filtered["sentence"].str.replace(r"\btidak\b", new_negation, case=False, regex=True)

    if save_path:
        df_filtered.to_csv(save_path, index=False)

negation_extraction(
    new_negation="nggak",
    save_path="smsa_filtered_nggak.csv"
)

negation_extraction(
    new_negation="gak",
    save_path="smsa_filtered_gak.csv"
)
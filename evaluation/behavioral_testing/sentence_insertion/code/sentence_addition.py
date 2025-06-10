import pandas as pd

def sentence_addition(addition, save_path):

    df = pd.read_csv("/path/to/smsa_dataset_changed_label.csv")

    df["sentence"] = df["sentence"] + addition

    if save_path:
        df[["sentence", "gold_label"]].to_csv(save_path, index=False)

sentence_addition(
    addition=" saya benci matematika .", # negative sentence
    save_path="smsa_added_neg.csv"
)

sentence_addition(
    addition=" saya cinta matematika .", # positive sentence
    save_path="smsa_added_pos.csv"
)
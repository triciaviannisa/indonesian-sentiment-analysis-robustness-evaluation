import pandas as pd

df = pd.read_csv("/path/to/smsa_dataset.tsv", sep="\t")

label_mapping = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}

df["gold_label"] = df["gold_label"].map(label_mapping)
df.to_csv("smsa_dataset_changed_label.csv", index=False)
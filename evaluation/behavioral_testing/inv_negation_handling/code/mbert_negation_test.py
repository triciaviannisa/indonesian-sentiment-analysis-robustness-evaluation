import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def negation_test(data_path, test_name, save_path=None):

    df = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained("./mbert_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased", num_labels=3)
    state_dict = torch.load("mbert_fine_tuned_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def predict(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).tolist()
        return predictions

    df["predicted_label"] = predict(df["sentence"].tolist())
    df["label_match"] = df["gold_label"] == df["predicted_label"]

    total = len(df)
    correct_behavior = df["label_match"].sum()
    failure_rate = ((total - correct_behavior) / total) * 100

    print(f"Behavioral Test Results ({test_name}):")
    print(f"Total samples: {total}")
    print(f"Failures (unexpected behavior): {total - correct_behavior}")
    print(f"Failure rate: {failure_rate:.2f}%")

    if save_path:
        df[["sentence", "gold_label", "predicted_label", "label_match"]].to_csv(save_path, index=False)

negation_test(
    data_path="/path/to/smsa_filtered_nggak.csv",
    test_name="tidak to nggak",
    save_path="result_tidak_to_nggak_mbert.csv"
)

negation_test(
    data_path="/path/to/smsa_filtered_gak.csv",
    test_name="tidak to gak",
    save_path="result_tidak_to_gak_mbert.csv"
)
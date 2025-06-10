import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def dir_test(model_name, tokenizer_path, model_path, state_path, save_path=None):

    df = pd.read_csv("/path/to/smsa_added_neg.csv")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    state_dict = torch.load(state_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def predict(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).tolist()
        return predictions

    df["predicted_label"] = predict(df["modified_sentence"].tolist())

    def expected_label(gold_label):
        if gold_label == 0:
            return {1, 2}
        elif gold_label == 1:
            return {2}
        elif gold_label == 2:
            return {2}
        else:
            return set()

    df["expected_label"] = df["gold_label"].apply(expected_label)

    df["label_match"] = df.apply(lambda row: row["predicted_label"] in row["expected_label"], axis=1)

    total = len(df)
    correct_behavior = df["label_match"].sum()
    failure_rate = ((total - correct_behavior) / total) * 100

    print(f"DIR Test Negative Sentence Results ({model_name}):")
    print(f"Total samples: {total}")
    print(f"Failures (unexpected behavior): {total - correct_behavior}")
    print(f"Failure rate: {failure_rate:.2f}%")

    if save_path:
        df[["sentence", "gold_label", "expected_label", "predicted_label", "label_match"]].to_csv(save_path, index=False)

dir_test(
    model_name="IndoBERT",
    tokenizer_path="./indobert_tokenizer",
    model_path="indobenchmark/indobert-base-p1",
    state_path="indobert_fine_tuned_model.pth",
    save_path="result_neg_indobert.csv"
)

dir_test(
    model_name="mBERT",
    tokenizer_path="./mbert_tokenizer",
    model_path="google-bert/bert-base-multilingual-cased",
    state_path="mbert_fine_tuned_model.pth",
    save_path="result_neg_mbert.csv"
)
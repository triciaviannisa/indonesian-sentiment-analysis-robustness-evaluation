import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def standard(model_name, tokenizer_path, model_path, state_path, save_path=None):

    df = pd.read_csv("/path/to/smsa_dataset_changed_label.csv")

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

    df["predicted_label"] = predict(df["sentence"].tolist())

    accuracy = accuracy_score(df["gold_label"], df["predicted_label"])
    precision = precision_score(df["gold_label"], df["predicted_label"], average="weighted")
    recall = recall_score(df["gold_label"], df["predicted_label"], average="weighted")
    f1 = f1_score(df["gold_label"], df["predicted_label"], average="macro")

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

standard(
    model_name="IndoBERT",
    tokenizer_path="./indobert_tokenizer",
    model_path="indobenchmark/indobert-base-p1",
    state_path="indobert_fine_tuned_model.pth",
    save_path="result_typo_indobert.csv"
)

standard(
    model_name="mBERT",
    tokenizer_path="./mbert_tokenizer",
    model_path="google-bert/bert-base-multilingual-cased",
    state_path="mbert_fine_tuned_model.pth",
    save_path="result_typo_mbert.csv"
)
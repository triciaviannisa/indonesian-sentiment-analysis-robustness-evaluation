import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def politeness(data_path, test_name, save_path=None):

    df = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained("./indobert_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
    state_dict = torch.load("indobert_fine_tuned_model.pth", map_location=torch.device("cpu"))
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

politeness(
    data_path="/path/to/high_politeness.csv",
    test_name="High Politeness",
    save_path="result_high_indobert.csv"
)

politeness(
    data_path="/path/to/medium_politeness.csv",
    test_name="Medium Politeness",
    save_path="result_medium_indobert.csv"
)

politeness(
    data_path="/path/to/low_politeness.csv",
    test_name="Low Politeness",
    save_path="result_low_indobert.csv"
)
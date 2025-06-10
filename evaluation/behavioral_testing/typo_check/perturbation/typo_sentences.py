import pandas as pd
import random

def add_indonesian_typos(sentence, typo_prob=0.3):
    words = sentence.split()
    perturbed_words = []

    for word in words:
        if len(word) > 3 and random.random() < typo_prob:
            typo_type = random.choice(["swap", "delete"])
            i = random.randint(1, len(word)-2)

            if typo_type == "swap":
                chars = list(word)
                chars[i], chars[i+1] = chars[i+1], chars[i]
                word = ''.join(chars)
            elif typo_type == "delete":
                word = word[:i] + word[i+1:]

        perturbed_words.append(word)

    return ' '.join(perturbed_words)

df = pd.read_csv("/path/to/smsa_dataset_changed_label.csv")
sentences = df["sentence"].tolist()

perturbed_sentences = [
    add_indonesian_typos(sentence) for sentence in sentences
]

df["sentence"] = perturbed_sentences
df.to_csv("smsa_typo_perturbation.csv", index=False)
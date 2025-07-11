# Evaluating the Robustness of Sentiment Analysis for Indonesian Using Behavioral Testing

## About
Repository for the materials associated with the Bachelor's Thesis "Evaluating the Robustness of Sentiment Analysis for Indonesian Using Behavioral Testing" at LMU Munich, 11th June, 2025.

## Dataset
- SmSA dataset from IndoNLU Benchmark
- SmSA test subset from the SmSA dataset, with the sentiment labels changed to numerics
- Self-constructed formality-level datasets

## Evaluation

### Conventional Metrics
- Accuracy
- Precision
- Recall
- Macro-averaged F-1 Score

### Behavioral Testing
- Formality Levels
- Negation Handling
- Orthographical Errors
- Sentence Insertion
- Scripts to run the behavioral testings
- Results of the behavioral testings in CSV files

## False Result Extraction
- Script to extract results where the predicted labels do not match the gold/expected labels
- Results of the extraction in CVS files

## Fine-tuning
- Google Colab noteboook for fine-tuning IndoBERT
- Google Colab noteboook for fine-tuning mBERT

## Python Dependencies
- Pandas
- NLTK
- Scikit-learn
- Random
- PyTorch
- Transformers

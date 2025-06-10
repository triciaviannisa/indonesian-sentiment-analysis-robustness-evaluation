# Bachelor's Thesis. Maria Patricia Viannisa. LMU, CIS. June 11, 2025.
Repository for the materials associated with the Bachelor's Thesis "Evaluating the Robustness of Sentiment Analysis for Indonesian Using Behavioral Testing"

## Dataset
- SmSA dataset from IndoNLU Benchmark
- SmSA test subset from the SmSA dataset, with the sentiment labels changed to numerics
- Self-constructed formality-level datasets

## Evaluation

### Conventional Metrics
- Accuracy
- Precision
- Recall
- F-1 Score

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
- Google Colab file for fine-tuning IndoBERT
- Google Colab file for fine-tuning mBERT

## Python Dependencies
- Pandas
- NLTK
- Scikit-learn
- Random
- PyTorch
- Transformers

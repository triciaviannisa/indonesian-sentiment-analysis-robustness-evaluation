import pandas as pd

def false_extraction(data_path, save_path):

    df = pd.read_csv(data_path)

    df_filtered = df[df["label_match"].astype(str).str.contains(r"\bFALSE\b", case=False, regex=True)]

    if save_path:
        df_filtered.to_csv(save_path, index=False)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/dir_test/results/result_neg_indobert.csv",
    save_path="dir_neg_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/dir_test/results/result_pos_indobert.csv",
    save_path="dir_pos_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/dir_test/results/result_neg_mbert.csv",
    save_path="dir_neg_mbert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/dir_test/results/result_pos_mbert.csv",
    save_path="dir_pos_mbert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/negation_handling/results/result_tidak_to_gak_indobert.csv",
    save_path="gak_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/negation_handling/results/result_tidak_to_nggak_indobert.csv",
    save_path="nggak_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/negation_handling/results/result_tidak_to_gak_mbert.csv",
    save_path="gak_mbert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/negation_handling/results/result_tidak_to_nggak_mbert.csv",
    save_path="nggak_mbert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/politeness_forms/results/result_high_indobert.csv",
    save_path="high_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/politeness_forms/results/result_medium_indobert.csv",
    save_path="medium_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/politeness_forms/results/result_low_indobert.csv",
    save_path="low_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/politeness_forms/results/result_high_mbert.csv",
    save_path="high_mbert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/politeness_forms/results/result_medium_mbert.csv",
    save_path="medium_mbert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/politeness_forms/results/result_low_mbert.csv",
    save_path="low_mbert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/typo_checker/results/result_typo_indobert.csv",
    save_path="typo_indobert_false.csv"
)

false_extraction(
    data_path="/Users/triciaviannisa/Thesis/code_testing/typo_checker/results/result_typo_mbert.csv",
    save_path="typo_mbert_false.csv"
)
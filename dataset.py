import pandas as pd


def read_excel_dataset_to_csv(excel_filepath, save=False) -> pd.DataFrame:
    filepath = "eval_dataset.csv"
    df = pd.read_excel(excel_filepath, skiprows=2)
    df.columns = ["book", "input", "expected_output", "retreived_context"]
    dataset = df[df["input"].notna()]

    if save:
        dataset.to_csv(filepath)
    
    return (dataset, filepath) if save else dataset



if __name__ == "__main__":
    x = read_excel_dataset_to_csv(
        excel_filepath='/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/Dataset for RAG Eval.xlsx', 
        save=True
    )
import pandas as pd


def load_dataset(path):
    """
    Loads the CSV dataset and returns a pandas DataFrame.
    """
    df = pd.read_csv(path)

    # Print info for inspection
    print("\n=== Dataset Loaded ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n=== First 5 Rows ===")
    print(df.head())

    # Show unique labels
    print("\n=== Unique Labels ===")
    print(df['labels'].unique())

    return df


if __name__ == "__main__":
    # Change path to your actual dataset file
    load_dataset("/home/adarshu/emotion-detector/data.csv")

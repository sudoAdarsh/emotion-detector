import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def load_and_clean(path):
    df = pd.read_csv(path)

    # Remove rows with empty text
    df = df.dropna(subset=['text'])

    # Remove rows with invalid labels (not between 0–7)
    df = df[df["labels"].between(0, 7)]

    # Strip spaces
    df["text"] = df["text"].str.strip()

    return df


def split_dataset(df):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["labels"]  # ensures balanced label distribution
    )
    return train_df, val_df


def tokenize_data(texts, tokenizer, max_len=128):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )


if __name__ == "__main__":
    DATASET_PATH = "/home/adarshu/emotion-detector/data.csv"

    print("=== Loading & Cleaning ===")
    df = load_and_clean(DATASET_PATH)
    print(df.head())

    print("\n=== Splitting Dataset ===")
    train_df, val_df = split_dataset(df)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    print("\n=== Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("\n=== Tokenizing Train Data ===")
    train_encodings = tokenize_data(train_df["text"], tokenizer)
    print({k: v.shape for k, v in train_encodings.items()})

    print("\n=== Tokenizing Validation Data ===")
    val_encodings = tokenize_data(val_df["text"], tokenizer)
    print({k: v.shape for k, v in val_encodings.items()})
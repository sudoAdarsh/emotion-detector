import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import EmotionDataset
from preprocess import load_and_clean, split_dataset, tokenize_data


def create_dataloaders(csv_path, batch_size=16):
    # 1. Load and clean
    df = load_and_clean(csv_path)

    # 2. Split
    train_df, val_df = split_dataset(df)

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 4. Tokenize
    train_enc = tokenize_data(train_df["text"], tokenizer)
    val_enc = tokenize_data(val_df["text"], tokenizer)

    # 5. Create Dataset objects
    train_dataset = EmotionDataset(train_enc, train_df["labels"].tolist())
    val_dataset = EmotionDataset(val_enc, val_df["labels"].tolist())

    # 6. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

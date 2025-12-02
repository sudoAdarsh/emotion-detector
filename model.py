import torch
from transformers import AutoModelForSequenceClassification

def load_model(num_labels=8):
    """
    Loads DistilBERT with a classification head for emotion detection.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    return model

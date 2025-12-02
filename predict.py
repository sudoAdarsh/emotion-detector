import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your trained model folder
MODEL_PATH = "emotion_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Make inference faster and disable gradient tracking
model.eval()

# Label mapping
id2label = {
    0: "joy",
    1: "anger",
    2: "fear",
    3: "sadness",
    4: "surprise",
    5: "disgust",
    6: "excitement",
    7: "neutral",
}

emoji_map = {
    "joy": "😊",
    "anger": "😠",
    "fear": "😨",
    "sadness": "😢",
    "surprise": "😲",
    "disgust": "🤢",
    "excitement": "🤩",
    "neutral": "😐",
}

def predict_emotion(text):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted label id
    pred_id = torch.argmax(logits, dim=1).item()

    # Convert to label and emoji
    emotion = id2label[pred_id]
    emoji = emoji_map[emotion]

    return {"emotion": emotion, "emoji": emoji}


# Testing the model
if __name__ == "__main__":
    text = input("Enter text: ")
    result = predict_emotion(text)
    print(result)

# Maps numeric IDs to emotion names
id2label = {
    0: "joy",
    1: "anger",
    2: "fear",
    3: "sadness",
    4: "surprise",
    5: "disgust",
    6: "excitement",
    7: "neutral"
}

# Maps emotion names back to numeric IDs
label2id = {label: id for id, label in id2label.items()}
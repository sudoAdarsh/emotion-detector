import torch
from torch.optim import AdamW
from transformers import get_scheduler

from dataloaders import create_dataloaders
from model import load_model

def train_model(csv_path, epochs=2, batch_size=16, lr=5e-5):

    # 1. Dataloaders
    train_loader, val_loader = create_dataloaders(csv_path, batch_size=batch_size)

    # 2. Load model
    model = load_model(num_labels=8)

    # 3. Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # 5. Learning rate scheduler
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 6. Training Loop
    model.train()
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print("Average training loss:", avg_loss)

        # Validation
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}  |  Accuracy: {accuracy:.4f}")

    # 7. Save final model
    model.save_pretrained("emotion_model")
    print("\nModel saved to ./emotion_model")



@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        total_loss += outputs.loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch["labels"]).sum().item()
        total += predictions.size(0)

    model.train()
    return total_loss / len(val_loader), correct / total

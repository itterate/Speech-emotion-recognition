import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNModel, Wav2VecModel
import config

def train_model(train_loader, val_loader, label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(len(label_map)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        for batch in train_loader:
            inputs, labels = batch["input_values"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    return model

import torch
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader, label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch["input_values"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    print(classification_report(true_labels, pred_labels, target_names=label_map.keys()))

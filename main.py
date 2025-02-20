from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
import config

if __name__ == "__main__":
    # Preprocess Data
    train_loader, val_loader, test_loader, label_map = preprocess_data(config.DATA_DIR)

    # Train Model
    model = train_model(train_loader, val_loader, label_map)

    # Evaluate Model
    evaluate_model(model, test_loader, label_map)

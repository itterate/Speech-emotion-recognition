import os
import numpy as np
import librosa
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import SpeechEmotionDataset
from transformers import Wav2Vec2Processor
import config

def extract_features(path):
    data, sample_rate = librosa.load(path, duration=2.5)

    # Feature extraction using librosa
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

    return np.hstack((zcr, mfcc, mel))

def preprocess_data(data_dir):
    # Load dataset and process audio files
    df = pd.DataFrame(columns=["file_path", "emotion"])
    
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            emotion = file.split("_")[2]  # Extract emotion from filename
            df = df.append({"file_path": os.path.join(data_dir, file), "emotion": emotion}, ignore_index=True)
    
    label_map = {label: idx for idx, label in enumerate(df["emotion"].unique())}
    df["emotion"] = df["emotion"].map(label_map)

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)
    
    train_dataset = SpeechEmotionDataset(train_df, processor)
    val_dataset = SpeechEmotionDataset(test_df, processor)
    test_dataset = SpeechEmotionDataset(test_df, processor)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, label_map

import torch
from torch.utils.data import Dataset
import librosa

class SpeechEmotionDataset(Dataset):
    def __init__(self, df, processor, max_length=32000):
        self.df = df
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]["file_path"]
        label = self.df.iloc[idx]["emotion"]

        speech, sr = librosa.load(audio_path, sr=16000)

        if len(speech) > self.max_length:
            speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), mode="constant")

        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values.squeeze(0)

        return {"input_values": inputs, "labels": torch.tensor(label, dtype=torch.long)}

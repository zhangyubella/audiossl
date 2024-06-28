import os

import torch.utils.data as data
import torchaudio


class custom_dataset(data.Dataset):
    def __init__(self, folder_path, split="train", subset=None, transform=None, target_transform=None, return_key = False, sr=16000):
        #self.db_path = db_path
        self.audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.flac')]
        self.return_key = return_key
        self.sr = sr
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, original_sample_rate = torchaudio.load(audio_path,

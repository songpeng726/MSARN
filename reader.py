import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise
# 加载并预处理音频
def load_audio(audio_path, mode='train', spec_len=128):
    # 读取音频数据
    wav, sr = librosa.load(audio_path, sr=44100)
    # spec_mag = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256)
    # if mode == 'train':
    #     crop_start = random.randint(0, spec_mag.shape[1] - spec_len)
    #     spec_mag = spec_mag[:, crop_start:crop_start + spec_len]
    # else:
    #     spec_mag = spec_mag[:, :spec_len]
    # mean = np.mean(spec_mag, 0, keepdims=True)
    # std = np.std(spec_mag, 0, keepdims=True)
    # spec_mag = (spec_mag - mean) / (std + 1e-5)
    # spec_mag = spec_mag[np.newaxis, :]
    # print(wav.shape)
    win_len = 176400
    wl = len(wav) - win_len
    win_start = random.randint(0, wl)
    win_data = wav[win_start: win_start+win_len]
    # maxamp = 0.
    # while maxamp < 0.005:
    #     win_start = random.randint(0, wl)
    #     win_data = wav[win_start: win_start + win_len]
    #     maxamp = np.max(np.abs(win_data))
    # print(win_data.shape)

    # win_data = wav[:win_len]
    # win_data = wgn(win_data, 5)
    # print(win_data.shape)
    wav = win_data.reshape(1, win_len) #原48000

    wav = wav.astype('float32')
    return wav


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, model='train'):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model


    def __getitem__(self, idx):
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        wav = load_audio(audio_path, mode=self.model)
        return wav, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)

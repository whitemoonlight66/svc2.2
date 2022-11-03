import os

import numpy as np
import torch

from data_utils import TextAudioSpeakerLoader
from utils import get_hparams_from_file
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text


dataset_path = "./dataset"  # 数据合集目录
file_train_name = "train.txt"
file_val_name = "val.txt"

train_f = open("./train.txt", "w", encoding="utf-8")
val_f = open("./val.txt", "w", encoding="utf-8")
hps_ms = get_hparams_from_file("configs/yilanqiu.json")


def get_spec_len(filename, hps_ms):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != hps_ms.data.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, hps_ms.data.sampling_rate))
    audio_norm = audio / hps_ms.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps_ms.data.filter_length,
                             hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                             center=False)
    spec = torch.squeeze(spec, 0)
    return spec.shape[-1]


with open(f"filelist/{file_train_name}", "r", encoding="utf-8") as f:
    file_list = f.readlines()
    for raw in file_list:
        wav_path = raw.split("|")[0]
        len_hbt = len(np.load(wav_path.replace("/wavs/", "/soft/")[:-4] + ".npy"))
        len_pt = get_spec_len(wav_path, hps_ms)
        if len_hbt == len_pt:
            train_f.write(raw)


train_f.close()
with open(f"filelist/{file_val_name}", "r", encoding="utf-8") as f:
    file_list = f.readlines()
    for raw in file_list:
        wav_path = raw.split("|")[0]
        len_hbt = len(np.load(wav_path.replace("/wavs/", "/soft/")[:-4] + ".npy"))
        len_pt = get_spec_len(wav_path, hps_ms)
        if len_hbt == len_pt:
            val_f.write(raw)

val_f.close()

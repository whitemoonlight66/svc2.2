import os
import subprocess

import numpy as np
import soundfile
import torch
import torchaudio

from mel_processing import spectrogram_torch
from utils import get_hparams_from_file
from utils import load_wav_to_torch


def format_wav(audio_path, tar_sample):
    raw_audio, raw_sample_rate = torchaudio.load(audio_path)
    if raw_sample_rate != tar_sample:
        tar_audio = torchaudio.transforms.Resample(orig_freq=raw_sample_rate, new_freq=tar_sample)(raw_audio)[0]
        soundfile.write(audio_path[:-4] + ".wav", tar_audio, tar_sample)


def mk_dir(path):
    for i in path:
        if not os.path.exists(i):
            os.mkdir(i)


def encode_hubert():
    cmd = 'python ./hubert/encode.py soft  %s/wavs %s/soft --extension .wav' % (real_path, real_path)
    p = subprocess.Popen(cmd, shell=True)
    return_code = p.wait()


def resize2d(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))

    return file_lists


def get_pitch(folder):
    wav_paths = get_end_file(f"{real_path}/wavs/{folder}", "wav")
    count = 0
    for wav_path in wav_paths:
        soft = np.load(wav_path.replace("wavs", "soft").replace(".wav", ".npy"))
        feature_pit = featureInput.compute_f0(wav_path)
        feature_pit = resize2d(feature_pit, soft.shape[0])
        pitch = featureInput.coarse_f0(feature_pit)
        np.save(wav_path.replace("wavs", "pitch").replace(".wav", ".npy"), pitch)
        count += 1
        print(f"pitch {folder}: {round(100 * count / len(wav_paths), 2)}%")


def make_filelist(folder):
    with open(f"{real_path}/raw_{folder}.txt", "w", encoding="utf-8") as f:
        for i in os.listdir(f"{real_path}/soft/{folder}"):
            f.write(
                f"{pre}/{dataset_name}/wavs/{folder}/{i.split('.')[0]}.wav{ID}{pre}/{dataset_name}/soft/{folder}/{i.split('.')[0]}.npy|{pre}/{dataset_name}/pitch/{folder}/{i.split('.')[0]}.npy\n")


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


def check_file(file_train_name):
    train_f = open(f"{real_path}/{file_train_name}.txt", "w", encoding="utf-8")
    with open(f"{real_path}/raw_{file_train_name}.txt", "r", encoding="utf-8") as f:
        file_list = f.readlines()
        count = 0
        for raw in file_list:
            wav_path = raw.split("|")[0]
            len_hbt = len(np.load(wav_path.replace("/wavs/", "/soft/")[:-4] + ".npy"))
            len_pt = get_spec_len(wav_path, hps_ms)
            if 0 <= len_pt - len_hbt * 2 <= 1:
                train_f.write(raw)
            count += 1
            print(f"check {file_train_name}: {round(100 * count / len(file_list), 2)}%")


if __name__ == "__main__":
    # 采样率，2.2必须为32000，自己别改
    target_sample = 32000
    hop_size = 320
    # 配置文件，这个nyarumul.json的speakers改成自己的，根目录建dataset，人物文件夹名完全对应这个
    hps_ms = get_hparams_from_file("nyarumul.json")
    for i in range(0, len(hps_ms["speakers"])):
        # speaker id
        ID = f"|{i}|"
        dataset_name = hps_ms["speakers"][i]  # speaker 文件夹名
        print(dataset_name)
        real_path = "./dataset/" + dataset_name
        pre = "dataset"
        # 转换采样率
        for w_path in get_end_file(real_path, "wav"):
            format_wav(w_path, target_sample)
        # hubert生成soft
        mk_dir([f"{real_path}/soft", f"{real_path}/soft/train", f"{real_path}/soft/val"])
        mk_dir([f"{real_path}/pitch", f"{real_path}/pitch/train", f"{real_path}/pitch/val"])
        encode_hubert()
        make_filelist("train")
        make_filelist("val")
        # pitch
        featureInput = FeatureInput(target_sample, hop_size)
        get_pitch("train")
        get_pitch("val")

        # 校对文件
        check_file("train")
        check_file("val")

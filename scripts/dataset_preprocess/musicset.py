import csv
import shutil
from pathlib import Path

import librosa
import pandas as pd
import torch
import torchaudio
import os
from dataset2lmdb import folder2lmdb, music2lmdb
from tqdm import tqdm

def parse_csv(csv_file,lbl_map,head_lines=3):
    dict = {}
    csv_lines = [ x.strip() for x in open(csv_file,"r").readlines()]
    for line in csv_lines[head_lines:]:
        line_list = line.split(",")
        id,label = line_list[0],line_list[3:]
        label = [x.strip(" ").strip('\"') for x in label]
        for x in label:
            if x == "":
                continue
            assert(x in lbl_map.keys())
        dict[id] = label
    return dict


def process(audio_path):
    tsv_output = [["files", "labels", "ext"]]
    all_files_count = 0
    for path in tqdm(Path(audio_path).rglob('*.[wav mp3 flac]*')):
        all_files_count += 1
        size = os.stat(path).st_size
        if  size < 500:
            print(f"Filtered {path} for its size({size}) less than 500.")
            continue
        base = path.name.split(".")[0]
        # 将所有label hardcode成ALL
        label = 'ALL'
        tsv_output.append([path, label, base])
    print(f"All files in {audio_path}: {all_files_count}")
    return tsv_output

def creatCSVFromAudioFolder(fpath, csv_path):
    import csv
    # 现在只有训练集是图书馆100多个小时的，需要遍历一遍生成csv
    folder_path = os.path.join(fpath, "unbalanced_train_segments/libraryUpload")
    fileIDs = []
    for file in Path(folder_path).glob('**/*.wav'):
        if file.is_file():
            fileIDs.append(file)
            #print(fileID)
    with open(os.path.join(csv_path, 'unbalanced_train_segments.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["# library CDs and VCDs", "", "", ""])
        writer.writerow(["# num_ytids=", "num_segs=", "num_unique_labels=1", "num_positive_labels=1"])
        writer.writerow(['ID', 'start_seconds', 'end_seconds', 'positive_labels'])
        for val in fileIDs:
            writer.writerow([val, '0', '', '\"ALL\"'])

def preprocess(audio_dir, csv_dir, split="train"):
    import os
    import json
    label_csv_file = os.path.join(csv_dir, "class_labels_indices.csv")
    lbl_map = {}
    with open(label_csv_file, "r") as f:
        label_lines = [x.strip() for x in f.readlines()]
    for line in label_lines[1:]:
        line_list = line.split(",")
        id, label = line_list[0], line_list[1]
        lbl_map[label] = int(id)
    with open(os.path.join(manifest_path_ub, "lbl_map.json"), "w") as f:
        json.dump(lbl_map, f)

    if split == "train":
        unb_train_audio_path = os.path.join(audio_dir, "unbalanced_train_segments")
        unb_train_tsv_output = process(unb_train_audio_path)
        with open(os.path.join(manifest_path_ub, "tr.tsv"), "w") as f:
            tsv_output = csv.writer(f, delimiter='\t')
            tsv_output.writerows(unb_train_tsv_output)
    elif split == "eval":
        eval_audio_path = os.path.join(audio_dir, "eval_segments")
        eval_tsv_output = process(eval_audio_path)
        with open(os.path.join(manifest_path_ub, "eval.csv"), "w") as f:
            tsv_output = csv.writer(f, delimiter='\t')
            tsv_output.writerows(eval_tsv_output)

def convert2mono(_audio_path):
    # 双声道转成单声道平均一首需要1.5s，fma_large数据集用了两天时间，保存
    from pydub import AudioSegment
    all_file_count = 0
    for path in tqdm(Path(_audio_path).rglob('*.[wav mp3 flac]*')):
        filepath = str(path)
        savepath = filepath.replace("fma_large_old", "fma_large")
        if os.path.isfile(savepath):
            all_file_count += 1
            continue
        try:
            stereo_audio = AudioSegment.from_file(path)
            mono_audio = stereo_audio.set_channels(1)
            mono_audio.export(savepath)
            all_file_count += 1
        except Exception:
            print(f"Error when converting file {filepath}.")
    print(f"Total file count in converted folder is {all_file_count}")

def copy_rename(audio_dir, audio_save_dir, csv_save_dir):
    filetypes = ['**/*.wav', '**/*.mp3', '**/*.flac']
    files = []
    for filetype in filetypes:
        files.extend(Path(audio_dir).glob(filetype))
    row_list = []
    i = 1  # 给曲目一个新的编号，并且保存对应关系
    for fullpath in files:
        audio_path = str(fullpath)
        audio_len = librosa.get_duration(filename=audio_path)
        sr = librosa.get_samplerate(path=audio_path)

        key = audio_path.removeprefix('/root/autodl-tmp/')
        new_audio_file = os.path.join(audio_save_dir, f"{i}.wav")
        row_list.append([key, new_audio_file,audio_len,sr])
        i = i + 1
        shutil.copy(audio_path, new_audio_file)
        print(f"move from {audio_path} to {new_audio_file}")
    df = pd.DataFrame(data=row_list, columns=['old_path', 'new_path', 'audio_len(s)', 'sr'])
    df.to_csv(os.path.join(csv_save_dir, 'metadata.csv'))

def split2segments(audio_dir, save_dir, stride=20, window_size=30):
    import time
    filetypes = ['**/*.wav', '**/*.mp3', '**/*.flac']
    files = []
    for filetype in filetypes:
        files.extend(Path(audio_dir).glob(filetype))
    i = 1
    start_time = time.time()
    for fullpath in files:
        audio_path = str(fullpath)
        audio_file = os.path.split(fullpath)[1]
        audio_len = librosa.get_duration(path=audio_path)
        waveform, sr = torchaudio.load(audio_path, normalize=True)
        # if sr != target_sr:
        #     #print(f"{audio_path} has the sampling sr {sr}hz, resampling to {target_sr}hz")
        #     waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        # waveform_mono = torch.mean(waveform, dim=0, keepdim=True)  # 双声道/立体声转成单声道

        upper = max(int(audio_len) - (stride - 1), 1)
        print(f"time spent: {time.time() - start_time}")
        for curr in range(0, upper, stride):
            start, end = curr, curr + window_size
            filename, file_extension = os.path.splitext(audio_file)
            new_filename = filename + f'_{start}s_{end}s' + file_extension
            # 保存切好的片段audio
            start_idx, end_idx = int(start * sr), int(end * sr)
            clip = waveform[:, start_idx: end_idx]
            print(f"before save time spent: {time.time() - start_time}")
            torchaudio.save(os.path.join(save_dir, new_filename), clip, sr)
            print(f"after save time spent: {time.time() - start_time}")
            #print(f"{new_filename} converted.")
        print(f"converted {i}/42447 ({audio_len}s), time spent: {time.time() - start_time}")
        i += 1


def get_metadata(audio_dir):
    total_duration = 0
    i = 1
    for file in os.listdir(audio_dir):
        if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
            filepath = os.path.join(audio_dir, file)
            dur = librosa.get_duration(path=filepath)
            sr = librosa.get_samplerate(filepath)
            total_duration += dur
            print(f"file {i} {file} samplerate: {sr}hz, {dur / 60.0}min")
            i += 1
        else:
            raise Exception(f"Not recognize file type: {file}.")
    print(f"total duration is {total_duration/3600.0}hours")
    return total_duration/3600.0

def resample_mono(audio_dir, save_dir, target_sr):
    import time
    filetypes = ['**/*.wav', '**/*.mp3', '**/*.flac']
    files = []
    for filetype in filetypes:
        files.extend(Path(audio_dir).glob(filetype))
    i = 1
    start_time = time.time()
    for fullpath in files:
        audio_path = str(fullpath)
        audio_file = os.path.split(fullpath)[1]
        save_path = os.path.join(save_dir, audio_file)
        if os.path.exists(save_path):
            print(f"{i}/42447 files already exists.")
            i += 1
        waveform, sr = torchaudio.load(audio_path, normalize=True)
        device = torch.device('cuda') # if torch.cuda.is_available() else 'cpu')
        waveform = waveform.to(device)
        if sr != target_sr:
            #print(f"{audio_path} has the sampling sr {sr}hz, resampling to {target_sr}hz")
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        torchaudio.save(save_path, waveform_mono.cpu(), target_sr)
        print(f"convert {i}/42447 file. Spent {time.time() - start_time}s")
        i += 1

def merge_lmdbs(lmdb_path):
    lmdb_paths = []
    for lmdb_data in os.listdir(lmdb_path):
        if lmdb_data.endswith('.lmdb'):
            lmdb_paths.append(os.path.join(lmdb_path, lmdb_data))


if __name__ == "__main__":
    SR = 22050
    data_path = "/20A021/dataset_from_dyl/train-50up"  # sys.argv[1:]
    audio_dir_old = os.path.join(data_path, "audio_old")
    audio_dir = os.path.join(data_path, "audio")
    audio_dir_split = os.path.join(data_path, "audio_split")
    csv_dir = os.path.join(data_path, "csv")

    lmdb_path_ub = os.path.join(data_path, "lmdb_path_ub")
    manifest_path_ub = os.path.join(data_path, "manifest_ub")

    #get_metadata(audio_dir=audio_dir)
    # resample以及单声道需要很长时间，因此只做这一次，保存为单声道、22050khz的音频。
    #resample_mono(audio_dir=audio_dir_old, save_dir=audio_dir, target_sr=SR)
    # save30秒audio花时间0.8s，那么2515小时需要
    #split2segments(audio_dir=audio_dir_old, save_dir=audio_dir_split, stride=20, window_size=30)

    # preprocess(audio_dir=audio_dir, csv_dir=csv_dir)
    # music2lmdb是将数据集剪成固定的长度，在这个过程中，写入lmdb容易内存溢出报错
    music2lmdb(data_path=data_path, split="train", stride=20, window_size=30,
               write_frequency=5000, max_num=100000)

    #folder2lmdb(manifest_path=manifest_path_ub,lmdb_path=lmdb_path_ub,split="train",
    #             max_len=30, write_frequency=5000, sr=SR)

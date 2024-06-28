import os
import pandas as pd
import soundfile as sf
from argparse import ArgumentParser

if __name__ == "__main__":
    # parser = ArgumentParser("Data Preprocess")
    # parser.add_argument("--root_path", default="/root/autodl-tmp/ccomhuqin/", type=str, required=True)
    # args = parser.parse_args()

    root_path = "/root/autodl-tmp/ccomhuqin/"
    # os.chdir(root_path)
    # if not os.path.exists("./train"):
    #     os.mkdir("./train")
    # if not os.path.exists("./eval"):
    #     os.mkdir("./eval")
    # mode = ["eval", "train"]
    # standard dcase label format: filename	onset	offset	event_label
    # for m in mode:
    #     full_meta_df = pd.read_csv(root_path+f"/meta/{m}/ccomhuqin_{m}.tsv", delimiter="\t")
    #     file_list = os.listdir(root_path+f"/data/{m}/")
    #     print(f"Total {m} files:", len(file_list))
    #     full_file_list = full_meta_df['filename'].values
    #     full_file_mask = [file in file_list for file in full_file_list]
    #     meta_df = full_meta_df[full_file_mask]
    #     meta_df.columns = ["filename", "onset", "offset", "event_label"]
    #     meta_df["filename"] = meta_df["filename"].map(lambda x: x + ".wav")
    #     meta_df.to_csv(f"./{m}/{m}.tsv", index=False, sep="\t")
    #     print(f"Total {m} files after filtering:", len(meta_df))

    # generate duration tsv for eval data
    eval_meta = pd.read_csv(root_path+"meta/test/eval.tsv", delimiter="\t")
    file_list = pd.unique(eval_meta["filename"].values)
    durations = []
    for file in file_list:
        wav, sr = sf.read(file)
        durations.append(min(10, len(wav) / sr))
    duration_df = pd.DataFrame({"filename": file_list, "duration": durations})
    duration_df.to_csv(root_path+"meta/test/eval_durations.tsv", index=False, sep="\t")
    print(f'save to {root_path}meta/test/eval_durations.tsv')


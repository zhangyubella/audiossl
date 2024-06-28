# NO USE
# # 将所有train_small.sh的参数复制到my_train_small.yaml中。不用sh文件了
import subprocess
import shlex

if __name__ == "__main__":
    PATH_DATASET = "/root/audiossl/audioset/manifest_ub"
    SAVE_PATH = "/root/autodl-tmp/savedir/pretrain/pretrain_0611"
    #subprocess.check_call(["./train_small.sh", PATH_DATASET, SAVE_PATH], shell=True)

    subprocess.run(['/bin/bash', './train_small.sh', PATH_DATASET, SAVE_PATH])

import subprocess

if __name__ == "__main__":
    # PATH_DATASET = "/root/audiossl/audioset/manifest_ub"
    # SAVE_PATH = "/root/autodl-tmp/savedir/"
    #subprocess.check_call(["./train_small.sh", PATH_DATASET, SAVE_PATH], shell=True)
    subprocess.run(['/bin/bash', './test_frame_atst.sh'])
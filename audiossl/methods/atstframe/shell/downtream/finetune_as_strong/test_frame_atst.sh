cd ../../../downstream
gpu_id='0,'
arch="frameatst"
lr_scale=1.0
bsz=64
max_epochs=40
lr="1e-1"
root_path="/root/autodl-tmp/savedir/finetune/frameatst/"
test_ckpt=root_path+"last.ckpt"
echo test: ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}

PATH+=:/root/miniconda3/envs/audiossl/bin
python train_as_strong.py --nproc ${gpu_id} --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "/root/autodl-tmp/savedir/pretrain/checkpoint-epoch= 59.ckpt" \
    --dcase_conf "./utils_as_strong/conf/frame_40.yaml" \
    --dataset_name "ccomhuqin" \
    --save_path root_path+"evaluation" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_frame_atst" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} \
    --test_from_checkpoint ${test_ckpt}

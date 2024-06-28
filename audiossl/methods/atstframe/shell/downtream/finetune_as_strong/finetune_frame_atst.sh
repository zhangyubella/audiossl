cd ../../../downstream
gpu_id='0'
arch="frameatst"
lr_scale=0.75
bsz=8
max_epochs=200

for lr in "1e-3"
do
    echo ${arch}, learning rate: ${lr}, lr_scale: ${lr_scale}
    PATH+=:/root/miniconda3/envs/audiossl/bin
    python train_as_strong.py --nproc ${gpu_id}, --learning_rate ${lr} --arch ${arch} \
    --pretrained_ckpt_path "/root/autodl-tmp/savedir/pretrain/checkpoint-epoch= 59.ckpt" \
    --dcase_conf "./utils_as_strong/conf/frame_40.yaml" \
    --dataset_name "ccomhuqin" \
    --save_path "/root/autodl-tmp/savedir/finetune/" \
    --batch_size_per_gpu ${bsz} \
    --prefix "_lr_${lr}_max_epochs_${max_epochs}" \
    --lr_scale ${lr_scale} \
    --max_epochs ${max_epochs} # \
    # \ --freeze_mode
done

以下记录了代码的修改和使用

一、 Pre-Train
把数据库文件夹放在/root/autodl-tmp/audioset/audio/unbalanced_train_segments/下面。运行audioset.py

二、 Finetune
1. 数据先处理成audioset的格式(scripts/dataset_preprocess/ccom_huqin)：
1）数据放在/root/autodl-tmp/ccomhuqin/下，先运行gen_dataset.py将audio和annotation分割成10s的片段，分割后的audio每十秒存成一个wav，所有的annotation保存
在一个表格下：“/root/autodl-tmp/ccomhuqin/meta/train/train.tsv” "/root/autodl-tmp/ccomhuqin/meta/eval/eval.tsv"
2）数据预处理：依次运行gen_tsv.py用于计算duration, common_label_filtrate.py过滤出训练集和eval集共有的label(按照目前分的train和eval，label都是共有的，因此不会过滤掉数据),
以及intersected_event_filtrate.py将同样event_label且在时间上重合的event合成一个，这些功能是从audioset_strong/拷贝修改。

2.
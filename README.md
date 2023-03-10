# Radical-Instance-Segmentation

# 文字部件分割的初步探索及优化研究

## 安装与配置
基于[Detectron2](https://github.com/facebookresearch/detectron2)开源工具箱。

## 数据集：
[**文字部件实例分割数据集**](https://github.com/Liutingjin/Radical-Instance-Segmentation/releases/download/datasets/radical2coco.zip)

## run：
`python train_net.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/SOLOv2_R50_3x`
## evaluation ：
`python train_net.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/SOLOv2_R50_3x \
    MODEL.WEIGHTS training_dir/SOLOv2_R50_3x/model_final.pth`

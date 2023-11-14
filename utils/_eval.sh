config_file_path='../adaptation_configs.yaml'

# 4.2	运行eval，打印测试结果 保存模型及阈值
python -m torch.distributed.launch --nproc_per_node=4 --master_port 6553 finetune/eval_resorting_torch.py --config=$config_file_path --val_batch_size 1 --img_size 224 --test_root  '/code/TCT_TRAIN/adaptation_end2end/data/lct_yolo_top1024_split_by_hospital/train.json' --model_name  'baseline_torch' --load_from_epoch 1

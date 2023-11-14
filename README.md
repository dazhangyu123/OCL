# Test-Time Training for Semantic Segmentation with Output Contrastive Loss

This is the Pytorch implementation of our "Test-Time Training for Semantic Segmentation with Output Contrastive Loss". This code is based on the paper [MaxSquare](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Domain_Adaptation_for_Semantic_Segmentation_With_Maximum_Squares_Loss_ICCV_2019_paper.pdf), due to using its pretrained checkpoint.


# Requirement
### Dataset
- Download [**Cityscapes**](https://www.cityscapes-dataset.com/), which contains 5,000 annotated images with 2048 × 1024 resolution taken from real urban street scenes. We use its validation set with 500 images. 
### Checkpoints
- Download the **[checkpoint](https://drive.google.com/open?id=1KP37cQo_9NEBczm7pvq_zEmmosdhxvlF&authuser=0)** pretrained on the GTA5 -> CityScapes task and place it in fold checkpoints.
- Download the **[checkpoint](https://drive.google.com/open?id=1wLffQRljXK1xoqRY64INvb2lk2ur5fEL&authuser=0)** pretrained on the SYNTHIA -> CityScapes task  and place it in fold checkpoints.

# Usage
### Baseline
```
python evaluate.py --pretrained_ckpt_file ./checkpoints/GTA5_source.pth --gpu 1 --method baseline --prior 0.0 --flip
python evaluate.py --pretrained_ckpt_file ./checkpoints/synthia_source.pth --gpu 1 --method baseline --prior 0.0 --flip
```
### Tri-TTT
```
python evaluate.py --pretrained_ckpt_file ./checkpoints/GTA5_source.pth --gpu 1 --method TTT --prior 0.85 --learning-rate 2e-5 --pos-coeff 3.0
python evaluate.py --pretrained_ckpt_file ./checkpoints/synthia_source.pth --gpu 1 --method TTT --prior 0.85 --learning-rate 1e-5 --pos-coeff 3.0
```
## Results

We present several transfered results reported in our paper.

**GTA2Cityscapes**

| Method  | Source only | OCL  | 
| :-----: |:-----------:|:----:|
| MIoUs |    37.5     | 45.0 |    

**Synthia2Cityscapes**

| Method  | Source only | OCL  | 
| :-----: |:-----------:|:----:|
| MIoUs |    31.5     | 36.9 |    


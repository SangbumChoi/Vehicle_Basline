# vehicle-ReID-baseline
## Introduction
Vehicle ReID baseline is a pytorch-based baseline for training and evaluating deep vehicle re-identification models on reid benchmarks.
## Updates
2020.07.28 updating how to deploy orientation

## Installation
1. cd to your preferred directory and run ' git clone https://github.com/Jakel21/vehicle-ReID '.
2. Install dependencies by pip install -r requirements.txt (if necessary).
## Datasets
+ [veri776](https://github.com/VehicleReId/VeRidataset)
+ [vehicleID](https://pkuml.org/resources/pku-vehicleid.html)

The keys to use these datasets are enclosed in the parentheses. See vehiclereid/datasets/__init__.py for details.Both two datasets need to pull request to the supplier.
Be careful of installing the repository of each datasets.

## Orientation and Keypoints
https://github.com/Pirazh/Vehicle_Key_Point_Orientation_Estimation
https://github.com/Zhongdao/VehicleReIDKeyPointData
according to personal CUDA memory allocation main script args have to be changed such as downsizing train batch size
```
python main.py --phase train --use_case stage1 --mGPU --lr 0.0001 --epochs 15 --train_batch_size 64
```

## Models
+ resnet50
## Losses
+ cross entropy loss
+ triplet loss

## Tutorial for my private setting
### train
Input arguments for the training scripts are unified in [args.py](./args.py).
To train an image-reid model with cross entropy loss, you can do
```
python train-xent-tri.py -s veri -t veri --height 128 --width 256 --optim amsgrad --lr 0.0003 --max-epoch 60 --stepsize 20 40 --train-batch-size 64 --test-batch-size 100 -a resnet50 --save-dir log/resnet50-veri --gpu-devices 0
```
### test
Use --evaluate to switch to the evaluation mode. In doing so, no model training is performed.
For example you can load pretrained model weights at path_to_model.pth.tar on veri dataset and do evaluation on VehicleID, you can do
```
python train_imgreid_xent.py -s veri -t vehicleID --height 128 --width 256 --test-size 800 --test-batch-size 100 --evaluate -a resnet50 --load-weights path_to_model.pth.tar --save-dir log/eval-veri-to-vehicleID --gpu-devices 0 
```

## Results
Some test results on veri776 and vehicleID:

### veri776
model:resnet50 

loss: xent+htri

| mAP | rank-1 | rank-5 | rank-20 |
|:---:| :----: | :----: | :-----: |
|59.0|87.6|94.3|98.2|

### vehicleID
model:resnet50 

loss: xent+htri

| testset size | mAP | rank-1 | rank-5 | rank-20 |
| :----------- |:---:| :----: | :----: | :-----: |
|800|76.4|69.1|85.8|94.5|
|1600|74.1|67.4|80.5|90.5|
|2400|71.4|65.2|78.3|89.2|



## Environments

- python 3
- torch = 1.4.0
- torchvision
- pyyaml
- Pillow
- numpy

## INSTALL

```
python setup.py develop --user
```

## Data Preparation
1. create dirs './dataset/coco2017
Data Structure is shown below.
```
coco2017
├── img
│   
│   
├── train.txt
│   
│   
└── test.txt
|
|__Instance_segmentation.json
   



## Train

For example:
python tools/train.py --config-file configs/trans10kv2/trans2seg/trans2seg_medium.yaml


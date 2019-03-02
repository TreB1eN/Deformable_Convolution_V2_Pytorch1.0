# Deformable Convolution V2 - MASK RCNN

Implementation of [Deformable Convolution V2](https://arxiv.org/abs/1811.11168) in Pytorch 1.0 codes

## Contributions

- The only repo that tries to reproduce the total set up of  [Deformable Convolution V2](https://arxiv.org/abs/1811.11168)  in full stable Pytorch codes
- The cuda codes are ported from MXNET to Pytorch, including Modulated Deformable Convolution and Modulated ROI Pooling , supporting stable pytorch1.0 version, gradient test code is provided
- Full training and test details are provided base on the framework of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)，Feature Mimicking branch is implemented

------

## Tips & Notice

- [train.py](https://github.com/TreB1eN/Deformable_Convolution_V2_Pytorch1.0/blob/master/train.py) only support single image per batch right now due to I don't have enough resource to run multi-batch and multi-cards training, but you can easily upgrade it to support batch and multi-card training if you check on the original [train_net.py](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/tools/train_net.py) from the original repo, because the framework supports everything.
- The training is ongoing, so the results and pretrained models will be published later due to it's really slow to train in my single card system.

## Difference with the paper

- Inspired by the idea from [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883), the model is trained from scratch, instead of finetuning from an Imagenet pretrained model
- Due to the same issue, Batchnorm is replaced by Groupnorm
- Weights for different branches are adjusted, and OHEM is used, compare to the original paper.

## How to use

### Preparation

#### Clone the this repository.

```shell
git clone git@github.com:TreB1eN/Deformable_Convolution_V2_Pytorch1.0.git
cd Deformable_Convolution_V2_Pytorch1.0/
```

#### Download the Pretrained models(Incoming)

### Installation 

1. Install PyTorch 1.0 and torchvision following the [official instructions](https://pytorch.org/).

2. Install dependencies

   ```shell
   pip install -r requirements.txt
   ```

3. Compile cuda ops.

   ```shell
   ./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
   ```

### Inference

```shell
Incoming
```

### Evaluate

```shell
Incoming
```

### Perform training on COCO dataset

For the following examples to work, you need to download the COCO dataset.
We recommend to symlink the path to the coco dataset to `datasets/` as follows

We use `minival` and `valminusminival` sets from [Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#coco-minival-annotations)

```bash
# symlink the coco dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
# for pascal voc dataset:
ln -s /path_to_VOCdevkit_dir datasets/voc
```

#### Create following folders 

```shell
Deformable_Convolution_V2
├── work_space
│   ├── model
│   ├── log
│   ├── final
│   ├── save
```

For training, just run:

```shell
python train.py 
```

All detailed configuration is in [configs/e2e_deformconv_mask_rcnn_R_50_C5_1x.yaml](https://github.com/TreB1eN/Deformable_Convolution_V2_Pytorch1.0/blob/master/configs/e2e_deformconv_mask_rcnn_R_50_C5_1x.yaml)

## References

[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

[paper](https://arxiv.org/abs/1811.11168) 

## Contact

Email : treb1en@qq.com

Questions and PRs are welcome, especially in helping me to get better trained models, due to I don't have enough resource to trained it sufficiently
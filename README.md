# function-network

This repository provides PyTorch implementation of our paper:

[基于关系图的三维场景预测与生成](http://qikan.cqvip.com/Qikan/Article/Detail?id=7108109785)

计算机辅助设计与图形学学报

## Prerequisites

* linux
* NVIDIA GPU + CUDA CuDNN
* Python 3.6

## Dependencies

Install python package dependencies through pip:

```
pip install -r requirements.txt
```

## Data

Please use [link](https://1drv.ms/u/s!Au7IAyHBmvyCiV10k9CJJdzzs-x9?e=R4gW3T
) to download the dataset and exact the file to dataset/data/ folder, e.g.

```
cd dataset/data
unzip data.zip
```

## Training

To train the pretrain model:

```
# pretrain the pointnet++
python pretrain_pointnet.py
```

To train the graph generation model:

```
# load the pretrain model, or you can just train directly
python train_graph.py --load_pretrain
```

To train the auto-encoder model:

```
# train part auto-encoder following multiscale strategy 16^3-32^3-64^3
sh scripts/easyscene/train_scene_partae_multiscale.sh # use two gpus 
```

To train the scene generation model:

```
# train seq2seq model
sh scripts/easyscene/train_scene_graph.sh
```

## Testing

You can find testing scripts in `proj_log/results` folder.

```
sh scripts/easyscene/rec_scene_graph.sh
```


# SeeMore

A pytorch implementation of the paper "SeeMore: bidirectional spatio-temporal predictive model from the knowledge-transfer perspective". The code is based on [PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning (TPAMI 2022)](https://github.com/thuml/predrnn-pytorch).



## Get Started

1. Install Python 3.6, PyTorch 1.9.0 for the main code. 

2. Download data. This repo contains code for two datasets: the [Moving Mnist dataset](https://cloud.tsinghua.edu.cn/d/21e9bde7cb954683ac94/) and the [KTH action dataset](https://cloud.tsinghua.edu.cn/d/7d19372a621a4952b738/).

3. Train and test the model. You can use the following bash script to train and test the model. The learned model will be saved in the `--save_dir` folder.

```
sh mnist_script/train_stage_1.sh
sh mnist_script/train_stage_2.sh
sh mnist_script/test.sh
```

<!-- 4. You can get **pretrained models** from  -->

# Randomness_Analysis for Data Augmentation


[![python](https://img.shields.io/badge/-Python_3.8_-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)


## 📌  Introduction 
**Randomness_Analysis for Data Augmentation**
[Paper] (Will be updated)

````

✅ This repo is about to data Augmentation containing cutmix, mixup, RandAugment ...

✅ You can do experiments with model ResNet50, ResNet101, WideResNet_28x10.

````


<br>


## 📌 How to use
### 1. Environments
```bash
# Create Conda Env
conda create -n myrand python=3.8

# Activate myrand
conda activate myrand

# cuda version
conda install cudatoolkit=11.3 -c pytorch
```



<br>




### 2. Data Preparation
Download [Tiny-imagenet](https://www.kaggle.com/c/tiny-imagenet)
```bash
# Make directory "data"
mkdir ./lightning-hydra-template/data

# unzip Tinyimagenet
unzip ./lightning-hydra-template/data/tiny-imagenet-200.zip

```
##### - Note that if you use Code related to CIFAR-100, then download the data automatically from torchvision.datasets


<br>


### 3. Setting Augmentations
If you want to use various Augmentations that we provide, you should change argument named "aug" at data_config and model_config.
Note that these are options about augmentations.
- [baseline, mixup, cutmix, randour_cutmix, cutmix_randour, randaug_mixup, randaug_cutmix, cutmix_randaug]





<br>


### 4. Training

##### - Note that these are options about {model_config_name}.
```
1. cifar100_resnet50
2. cifar100_resnet101
3. cifar100_wideresnet

4. tinyimagenet_resnet50
5. tinyimagenet_resnet101
6. tinyimagenet_wideresnet
```

##### - Note that these are options about {data_config_name}.
```
1. cifar100
2. tinyimagenet
```


#### command
```bash
# train on CPU
python train.py trainer=cpu model={model_config_name} data={data_config_name}

# train on 1 GPU
python train.py trainer=gpu model={model_config_name} data={data_config_name}

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer=ddp trainer.devices=4 model={model_config_name} data={data_config_name}

```


## 📌 Our results
### 1. Base Augmentation
#### TinyImagenet
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---|
|Wide_Resnet_28x10|	0.6662	|TinyImagenet|	Baseline|
|Wide_Resnet_28x10|	0.6686	|TinyImagenet|	Cutout	|
|Wide_Resnet_28x10|	0.6889	|TinyImagenet|	mixup|	
|Wide_Resnet_28x10|	**0.7057**	|TinyImagenet|	Cutmix|	
|Wide_Resnet_28x10|	0.687|	TinyImagenet|	Randaugment	|


<br>


#### CIFAR-100
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---|
|Wide_Resnet_28x10	|0.7943	|Cifar100|	Baseline|
|Wide_Resnet_28x10|	0.8096|	Cifar100	|Cutout|
|Wide_Resnet_28x10|	0.8232|	Cifar100	|mixup|
|Wide_Resnet_28x10|	**0.8237**|	Cifar100|	Cutmix|
|Wide_Resnet_28x10|	0.8089|	Cifar100	|Randaugment|




<br>

### 2. RandAugment + mixup, cutmix
#### TinyImagenet
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10	|**0.7041**	|TinyImagenet|	Randaugment → mixup|
|Wide_Resnet_28x10	|**0.716**	|TinyImagenet|	Randaugment → cutmix|
|Wide_Resnet_28x10	|0.6889|	TinyImagenet|	mixup|
|Wide_Resnet_28x10	|0.7057	|TinyImagenet	|cutmix|
|Resnet50|	**0.5871**|	TinyImagenet	|Randaugment → mixup|
|Resnet50|	**0.639**	|TinyImagenet	|Randaugment → cutmix|
|Resnet50|	0.5298	|TinyImagenet|	mixup|
|Resnet50	|0.5768|	TinyImagenet|	cutmix|
|Resnet101|	**0.5912**|	TinyImagenet|	Randaugment → mixup|
|Resnet101|	**0.6362**|	TinyImagenet|	Randaugment → cutmix|
|Resnet101|	0.5789|	TinyImagenet|	mixup|
|Resnet101|	0.5285|	TinyImagenet	|cutmix|

<br>


#### CIFAR-100
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10	|**0.8325**|	Cifar100	|Randaugment + mixup|
|Wide_Resnet_28x10	|**0.8358**	|Cifar100|	Randaugment + cutmix|
|Wide_Resnet_28x10|	0.8232	|Cifar100	|mixup|
|Wide_Resnet_28x10|	0.8237	|Cifar100|	cutmix|
|Resnet50|	**0.6485**	|Cifar100	|Randaugment + mixup
|Resnet50|	**0.6803**	|Cifar100	|Randaugment + cutmix|
|Resnet50|	0.6262	|Cifar100|	mixup|
|Resnet50	|0.6453	|Cifar100	|cutmix|
|Resnet101|	**0.6431**|	Cifar100	|Randaugment + mixup|
|Resnet101|	**0.6596**	|Cifar100|	Randaugment + cutmix|
|Resnet101|	0.619|	Cifar100|	mixup|
|Resnet101|	0.6382	|Cifar100	|cutmix|


<br>

### 3. RandAugment vs RandOur
#### TinyImagenet
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10|	0.7163	|TinyImagenet|	Randaugment → cutmix|
|Wide_Resnet_28x10|	**0.7201**|	TinyImagenet|	Randours → cutmix|
|Resnet50|	**0.639**|	TinyImagenet|	Randaugment → cutmix|
|Resnet50|	0.5989	|TinyImagenet|	Randours → cutmix|
|Resnet101|	**0.6362**	|TinyImagenet|	Randaugment → cutmix|
|Resnet101	|0.6	|TinyImagenet|	Randours → cutmix|


<br>


#### CIFAR-100
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10|	0.8358|	Cifar100	|Randaugment → cutmix|
|Wide_Resnet_28x10|	**0.8432**|	Cifar100|	Randours → cutmix|
|Resnet50|	**0.685**|	Cifar100|	Randaugment → cutmix|
|Resnet50|	0.6536|	Cifar100	|Randours → cutmix|
|Resnet101|	**0.6596**	|Cifar100|Randaugment → cutmix|
|Resnet101	|0.6517	|Cifar100|	Randours → cutmix|


<br>



### 4. Randomness + Cutmix vs Cutmix + Randomness
#### TinyImagenet
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10|	**0.7163**|	TinyImagenet|	Randaugment → cutmix|
|Wide_Resnet_28x10|	0.7015|	TinyImagenet|	cutmix → Randaugment|
|Resnet50|	**0.639**|	TinyImagenet|	Randaugment → cutmix|
|Resnet50	|0.5807|	TinyImagenet|	cutmix → Randaugment|
|Resnet101|	**0.6362**|	TinyImagenet|	Randaugment → cutmix|
|Resnet101	|0.5835|	TinyImagenet	|cutmix → Randaugment|

<br>

|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10|	**0.7201**|	TinyImagenet	|Randours → cutmix|
|Wide_Resnet_28x10|	0.7173|	TinyImagenet	|cutmix → Randours |
|Resnet50|	**0.5989**	|TinyImagenet|	Randours → cutmix|
|Resnet50|	0.5979|	TinyImagenet|	cutmix → Randours |
|Resnet101	|**0.6**|TinyImagenet	|Randours → cutmix|
|Resnet101	|0.5967	|TinyImagenet	|cutmix → Randours |

<br>

#### CIFAR-100
|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10	|**0.8358**|Cifar100|	Randaugment → cutmix|
|Wide_Resnet_28x10|	0.8253|	Cifar100|	cutmix → Randaugment|
|Resnet50	|**0.685** |	Cifar100	|Randaugment → cutmix|
|Resnet50	|0.6141|	Cifar100	|cutmix → Randaugment|
|Resnet101	|**0.6596**	|Cifar100	|Randaugment → cutmix|
|Resnet101	|0.6117|	Cifar100	|cutmix → Randaugment|

<br>

|Model|Accuracy	|Dataset|Augmentation|
|--- |---| ---| ---| 
|Wide_Resnet_28x10	|**0.843**|	Cifar100	|Randours → cutmix|
|Wide_Resnet_28x10	|0.84|	Cifar100|	cutmix → Randours |
|Resnet50|	**0.6536**|Cifar100|	Randours → cutmix|
|Resnet50	|0.6516	|Cifar100	|cutmix → Randours |
|Resnet101|	**0.6517**|	Cifar100|	Randours → cutmix|
|Resnet101|	0.6438	|Cifar100	|cutmix → Randours |

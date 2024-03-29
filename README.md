[![arXiv](https://img.shields.io/badge/arXiv-2206.09843-b31b1b.svg)](https://arxiv.org/abs/2206.09843)

Official pytorch implementation of the paper:

*"Contextual Squeeze-and-Excitation for Efficient Few-Shot Image Classification"* (2022) Patacchiola, M., Bronskill, J., Shysheya, A., Hofmann, K., Nowozin, S., Turner R.E., *Advances in Neural Information Processing (NeurIPS)* [[arXiv]](https://arxiv.org/abs/2206.09843)


```bibtex
@inproceedings{patacchiola2022contextual,
  title={Contextual Squeeze-and-Excitation for Efficient Few-Shot Image Classification},
  author={Patacchiola, Massimiliano and Bronskill, John and Shysheya, Aliaksandra and Hofmann, Katja and Nowozin, Sebastian and Turner, Richard E},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

**Overview** Recent years have seen a growth in user-centric applications that require effective knowledge transfer across tasks in the low-data regime. An example is personalization, where a pretrained system is adapted by learning on small amounts of labeled data belonging to a specific user. This setting requires high accuracy under low computational complexity, therefore the Pareto frontier of accuracy vs. adaptation cost plays a crucial role. In this paper we push this Pareto frontier in the few-shot image classification setting with a key contribution: a new adaptive block called Contextual Squeeze-and-Excitation (CaSE) that adjusts a pretrained neural network on a new task to significantly improve performance with a single forward pass of the user data (context). We use meta-trained CaSE blocks to conditionally adapt the body of a network and a fine-tuning routine to adapt a linear head, defining a method called UpperCaSE. UpperCaSE achieves a new state-of-the-art accuracy relative to meta-learners on the 26 datasets of VTAB+MD and on a challenging real-world personalization benchmark (ORBIT), narrowing the gap with leading fine-tuning methods with the benefit of orders of magnitude lower adaptation cost.

<p align="center">
  <img width="550" alt="Squeeze-and-Excitation (SE) vs Contextual Squeeze-and-Excitation (CaSE)" src="./etc/model_with_caption.png">
</p>

Requirements
------------

If you are interested in **meta-training** the model from scratch you need the following packages:

- Python >= 3.7
- PyTorch >= 1.8
- TensorFlow >= 2.3 (for Meta-Dataset and VTAB)
- TensorFlow Datasets >= 4.3 (for VTAB) [[link](https://www.tensorflow.org/datasets)]
- Gin Config >= 0.4 (for Meta-Dataset)

We also provide a conda environment (see instructions below).

If you are interested in using the model for **inference only** (on a dataset of your choice) you just need Pytorch and common libraries (e.g. Numpy). See the example script here [example.py](./example.py).


Installation
-------------

**Training on MetaDataset** If you want to train the model on MetaDataset you need to download and prepare the dataset, please follow the instructions reported here:

- https://github.com/google-research/meta-dataset
- https://github.com/cambridge-mlg/LITE

**Evaluation on VTAB** If you want to evaluate the model on VTAB you need to download and prepare the dataset. Please follow the instructions reported here:

- https://github.com/google-research/task_adaptation

**Installation via Conda** We provide a file called `environment.yml` that you can use to install the conda environment. This can be done with the following command:

```
conda env create -f environment.yml
```

This will create an environment called `myenv` that you will need to activate via `conda activate myenv`.

**Pretrained models** We have included a pretrained model in `./checkpoints/UpperCaSE_CaSE64_min16_EfficientNetB0.dat`. This is a pretrained EfficientNetB0 with CaSE blocks (reduction 64, min-clip 16), which is the same reported in the paper. This can be directly used for evaluation on MetaDataset and VTAB without the need for meta-training.

For the pretrained ResNet50-S you need to download the model from the [Big Transfer repository](https://github.com/google-research/big_transfer) as follows:

```
wget wget https://storage.googleapis.com/bit_models/BiT-S-R50x1.npz
```


Generic Usage
-------------

Our pretrained model can be easily used on a dataset of your choice. If you want to use the model only for inference (no training on MetaDataset) then you just need to install Pytorch.

We provide an example script that runs the pretrained UpperCaSE (with EfficientNetB0) for inference on CIFAR100 and SVHN in the file [example.py](./example.py).


Adding CaSE layers to your custom model
----------------------------------------

The CaSE block is a self-contained module that you can find under [adapters/case.py](./adapters/case.py). You can import and use a CaSE block like any other layer in a sequetial model in Pytorch. We reccomend to place CaSE layers after the activation function (e.g. ReLU, SiLU, etc) as follows:

```python
import torch
from case import CaSE

model = torch.nn.Sequential(
          torch.nn.Conv2d(128, 32, kernel_size=3),
          torch.nn.ReLU(),
          CaSE(cin=32, reduction=64, min_units=16)
          torch.nn.Conv2d(32, 64, kernel_size=3),
          torch.nn.ReLU()
          CaSE(cin=64, reduction=64, min_units=16)
        )
```

CaSE layers can be trained following this procedure:

1. Train a backbone with a standard supervised-learning routine. It is possible to use Pytorch models pretrained on ImageNet.
2. Add a set of CaSE layers in the backbone (see the paper for more details about this step).
3. Meta-train the parameters of the CaSE layers (keep frozen the parameters of the backbone).

Step 3 can be easily performed by isolating the learnable parameters of CaSE and passing them to the optimizer as follows:

```python
import torch
from case import CaSE

# "backbone" is a pretrained neural net containing CaSE layers

params_list = list()
for module_name, module in backbone.named_modules():
    for parameter in module.parameters():
        if(type(module) is CaSE): params_list.append(parameter)
        
optimizer = torch.optim.Adam(params_list, lr=0.001)
```


Reproducing the experiments
---------------------------

To reproduce the results of the paper you need to have installed MetaDataset and VTAB as explained above. After you have done this, follow the instructions below.

1. MetaDataset requires the following command to be run before every simulation:

```
ulimit -n 50000
```

```
export META_DATASET_ROOT=<root directory of the Meta-Dataset repository>
```

2. For training UpperCaSE on MetaDataset (and testing at the end) use the following command (replacing with the appropriate paths on your system):

```
python run_metadataset.py --model=uppercase --backbone=EfficientNetB0 --data_path=/path_to_metadataset_records --log_path=./logs/uppercase_EfficientNetB0_seed1_`date +%F_%H%M%S`.csv --image_size=224 --num_test_tasks=1200 --mode=train_test
```

The log-file will be saved in `./log`. Change the backbone type or image size if you want to try other configurations. Available backbones are: `["BiT-S-R50x1", "ResNet18", "EfficientNetB0"]`.

3. For testing on MetaDataset use the following command (replacing with the appropriate paths on your system):

```
python run_metadataset.py --model=uppercase --backbone=EfficientNetB0 --data_path=/path_to_metadataset_records --log_path=./logs/uppercase_EfficientNetB0_seed1_`date +%F_%H%M%S`.csv --image_size=224 --num_test_tasks=1200 --mode=test --resume_from=/path_to_checkpoint
```

4. The MetaDataset results saved in the log file can be printed in a nice way using the `printer.py` by running: 

```
python printer.py --log_path=./logs/name_of_your_log_gile.csv
```

5. For evaluation ov VTAB you need first to train on MetaDataset and then use the saved checkpoint. Use this command for evaluation (replacing with the appropriate paths on your system):

```
python run_vtab.py --model=uppercase --backbone=EfficientNetB0 --download_path_for_tensorflow_datasets=/path_to_tensorflow_datasets --log_path=./logs/vtab_UpperCaSE_EfficientNetB0_`date +%F_%H%M%S`.csv --image_size=224 --batch_size=50 --download_path_for_sun397_dataset=/path_to_sun397_images --resume_from=/path_to_checkpoint
```

Results are saved in the `./logs` folder as CSV files.

License
-------

MIT License

Copyright (c) 2022 The authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


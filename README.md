This repository includes the code of the paper `"Contextual Squeeze-and-Excitation for Efficient Few-Shot Image Classification"`.

Requirements
------------

- Python >= 3.7
- PyTorch >= 1.8
- TensorFlow >= 2.3 (for Meta-Dataset and VTAB)
- TensorFlow Datasets >= 4.3 (for VTAB) [[link](https://www.tensorflow.org/datasets)]
- Gin Config >= 0.4 (for Meta-Dataset)

Installation
-------------

The code requires the installation of MetaDataset and VTAB. Please follow the instructions reported here:

- https://github.com/google-research/meta-dataset
- https://github.com/cambridge-mlg/LITE

Usage
-----

**Note**: we have included a pretrained model in the `/checkpoints` folder, EfficientNetB0 with CaSE blocks (reduction 64, min-clip 16), which is the same reported in the paper. This can be directly used for evaluation on MetaDataset and VTAB without the need for meta-training.

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

The log-file will be saved in `./log`. Change the backbone type or image size if you want to try other configurations. Available backbones are: `["BiT-S-R50x1", "ResNet18", "ResNet50", "EfficientNetB0"]`.

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

Results are saved in the `./logs` folder.



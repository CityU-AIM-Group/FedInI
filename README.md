# FedInI
[MICCAI' 22] Intervention &amp; Interaction Federated Abnormality Detection with Noisy Clients


By Xinyu Liu

## Installation

Check [FCOS](https://github.com/tianzhi0549/FCOS/blob/master/INSTALL.md) for installation instructions.

## Data preparation

Step 1: Download the GLRC dataset as well as the box annotations from [this URL](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FCBUOR), split the GLRC subset and convert to VOC format. Use the provided [client split](https://github.com/CityU-AIM-Group/FedInI/blob/main/ImageSets) to replace ImageSets dir.

```
[DATASET_PATH]
└─ GLRC
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
```

Step 2: Generating class-conditioned noisy annotations with [this script](https://github.com/CityU-AIM-Group/FedInI/blob/main/generate_noise_ccd.py) or instance-dependant noise with [this script](https://github.com/CityU-AIM-Group/FedInI/blob/main/generate_noise_ccd.py)

Step 3: change the data root for your dataset at [paths_catalog.py](https://github.com/CityU-AIM-Group/FedInI/blob/main/fcos_core/config/paths_catalog.py).


## Get start 

Train with FedInI:
(Our code currently only supports single-GPU training.)

```
python tools/train_net.py --config ./configs/federated/glrc.yaml SOLVER.ANNOTATIONS 0.3 OUTPUT_DIR output_fedini SOLVER.METHOD att
```

As a comparison, train with FedAvg:
```
python tools/train_net.py --config ./configs/federated/glrc.yaml SOLVER.ANNOTATIONS 0.3 OUTPUT_DIR output_fedavg SOLVER.METHOD ori
```

 
## Citation 

If you think this work is helpful for your project, kindly give it a star and citation:
```
@inproceedings{liu2022intervention,
  title={Intervention \& Interaction Federated Abnormality Detection with Noisy Clients},
  author={Liu, Xinyu and Li, Wuyang and Yuan, Yixuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={309--319},
  year={2022},
  organization={Springer}
}
```

## Acknowledgements

The work is based on [FCOS](https://github.com/tianzhi0549/FCOS).
 
## Contact 

If you have any problems, please feel free to contact me at xliu423-c@my.cityu.edu.hk. Thanks.

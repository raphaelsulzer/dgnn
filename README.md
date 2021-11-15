# Scalable Surface Reconstruction with Delaunay-Graph Neural Networks
[**Paper**](https://arxiv.org/pdf/2107.06130.pdf) | [**Video**](https://youtu.be/KIrCDGhS10o) <br>

This repository contains the implementation of the paper:

Scalable Surface Reconstruction with Delaunay-Graph Neural Networks<br />
Raphael Sulzer, Loic Landrieu, Renaud Marlet, Bruno Vallet<br />
[**SGP 2021**](https://sgp2021.github.io/program/)  

If you find our code or paper useful, please consider citing
```bibtex
@article{2021,
   title={Scalable Surface Reconstruction with Delaunay‐Graph Neural Networks},
   volume={40},
   ISSN={1467-8659},
   url={http://dx.doi.org/10.1111/cgf.14364},
   DOI={10.1111/cgf.14364},
   number={5},
   journal={Computer Graphics Forum},
   publisher={Wiley},
   author={Sulzer, R. and Landrieu, L. and Marlet, R. and Vallet, B.},
   year={2021},
   month={Aug},
   pages={157–167}
}
```


## Installation

Please follow the instructions step-by-step.

1. Clone the repository to your local machine and enter the folder
```
git clone git@github.com:raphaelsulzer/dgnn.git
cd dgnn
```

2. Create an anaconda environment called `dgnn`
```
conda env create -f environment.yaml
conda activate dgnn
```

3. Compile the extension module `libmesh` (taken from [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks) [1])
```
cd utils
python setup_libmesh_convonet.py build_ext --inplace
```

## Mesh reconstruction using our pretrained model

### Berger et al. dataset

Reconstruct the Berger et al. [2] dataset from the scans used in the paper.

1. Download and unzip the dataset in the `data` folder

```
cd data
bash download_reconbench.sh
```

2. Reconstruct and evaluate the meshes

```
python run.py -i --config configs/reconbench.yaml
```

### ETH3D dataset

Reconstruct all training scenes of the ETH3D [3] dataset from the MVS point clouds used in the paper.

1. Download and unzip the dataset in the `data` folder

```
cd data
bash download_eth3d.sh
```

2. Reconstruct the meshes

```
python run.py -i --config configs/eth3d.yaml
```

To evaluate the results you can e.g. sample points on the meshes and use the [multi-view-evaluation](https://github.com/ETH3D/multi-view-evaluation) tool provided by the ETH3D dataset authors.

### Custom object or scene

To reconstruct any object or scene from a point cloud you need a `pointcloud.npz` file.
See `Custom dataset` for the structure of the file. Once you have the `pointcloud.npz` file run.

```
python run.py -i --config configs/default.yaml --file path/to/pointcloud.npz
```



## Training

Training a new model requires closed ground truth meshes. You can use the procedure explained 
in `Custom dataset` to construct a new dataset from such meshes. Once you have created the dataset you need to
add your dataset to the `getDataset()` function in `processing/dataset.py` and
prepare a `custom.yaml` file (see `configs/reconbench.yaml` for an example). 
The `.yaml` file also allows to change several model 
parameters.

To train a new model run

```
python run.py -t --config configs/custom.yaml
```


[comment]: <> (starting from an existing mesh &#40;see `scan`&#41; or from a)
[comment]: <> (`pointcloud.npz` file &#40;see `feat`, omit `-g` flag&#41;.)

## Custom dataset

To train or reconstruct a custom dataset you need to provide for each object or scene a `pointcloud.npz` file
containing a `points`, `normals` and `sensor_position` field (see `data/reconbench/3_scan/` for examples of
such files). The `normals` field is not actually used in this work but has to be present in the file. 
It can e.g. be set to contain zero vectors. The datatype of all fields has to be `np.float64`.
Furthermore you need several files containing the 3D Delaunay triangulation and feature information for each object
or scene.

We provide prebuild binaries for Ubuntu 18.04 to synthetically scan a mesh, build a 3D Delaunay triangulation,
and extract ground truth labels and features.

1. Scan ground truth mesh

```
cd utils
./scan -w path/to/working_directory -i "" -g path/to/groundtruth_mesh/from/working_directory
```

Note that this uses our own scanning procedure and not the one from Berger et al. [2] 
used in the paper.

2. Extract labels and features

```
cd utils
./feat -w path/to/working_directory -i "" -g path/to/groundtruth_mesh/from/working_directory
```

See e.g. `processing/reconbench/feat.py` and `processing/reconbench/scan.py` for examples
to batch process all files in your dataset.


## References

[1] PENG S., NIEMEYER M., MESCHEDER L., POLLEFEYS M.,
GEIGER A.: Convolutional occupancy networks. In ECCV (2020).

[2] BERGER M., LEVINE J. A., NONATO L. G., TAUBIN G.,
SILVA C. T.: A benchmark for surface reconstruction. ACM Transaction on Graphics. (2013).

[3] SCHÖPS T., SCHÖNBERGER J. L., GALLIANI S., SATTLER
T., SCHINDLER K., POLLEFEYS M., GEIGER A.: A multi-view stereo
benchmark with high-resolution images and multi-camera videos. In
CVPR (2017).
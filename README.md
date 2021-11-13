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

2. Create an anaconda environment called `dgnn` using
```
conda env create -f environment.yaml
conda activate dgnn
```

3. Compile the extension module `libmesh` (taken from [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks))
```
cd utils
python setup_libmesh_convonet.py build_ext --inplace
```

[comment]: <> (## Demo)

[comment]: <> (First, run the script to get the demo data:)

[comment]: <> (```)

[comment]: <> (bash scripts/download_demo_data.sh)

[comment]: <> (```)

[comment]: <> (### Reconstruct Large-Scale Matterport3D Scene)

[comment]: <> (You can now quickly test our code on the real-world scene shown in the teaser. To this end, simply run:)

[comment]: <> (```)

[comment]: <> (python generate.py configs/pointcloud_crop/demo_matterport.yaml)

[comment]: <> (```)

[comment]: <> (This script should create a folder `out/demo_matterport/generation` where the output meshes and input point cloud are stored.)

[comment]: <> (**Note**: This experiment corresponds to our **fully convolutional model**, which we train only on the small crops from our synthetic room dataset. This model can be directly applied to large-scale real-world scenes with real units and generate meshes in a sliding-window manner, as shown in the [teaser]&#40;media/teaser_matterport.gif&#41;. More details can be found in section 6 of our [supplementary material]&#40;http://www.cvlibs.net/publications/Peng2020ECCV_supplementary.pdf&#41;. For training, you can use the script `pointcloud_crop/room_grid64.yaml`.)


[comment]: <> (### Reconstruct Synthetic Indoor Scene)

[comment]: <> (<div style="text-align: center">)

[comment]: <> (<img src="media/demo_syn_room.gif" width="600"/>)

[comment]: <> (</div>)

[comment]: <> (You can also test on our synthetic room dataset by running: )

[comment]: <> (```)

[comment]: <> (python generate.py configs/pointcloud/demo_syn_room.yaml)

[comment]: <> (```)

[comment]: <> (## Dataset)

[comment]: <> (To evaluate a pretrained model or train a new model from scratch, you have to obtain the respective dataset.)

[comment]: <> (In this paper, we consider 4 different datasets:)

[comment]: <> (### ShapeNet)

[comment]: <> (You can download the dataset &#40;73.4 GB&#41; by running the [script]&#40;https://github.com/autonomousvision/occupancy_networks#preprocessed-data&#41; from Occupancy Networks. After, you should have the dataset in `data/ShapeNet` folder.)

[comment]: <> (### Synthetic Indoor Scene Dataset)

[comment]: <> (For scene-level reconstruction, we create a synthetic dataset of 5000)

[comment]: <> (scenes with multiple objects from ShapeNet &#40;chair, sofa, lamp, cabinet, table&#41;. There are also ground planes and randomly sampled walls.)

[comment]: <> (You can download our preprocessed data &#40;144 GB&#41; using)

[comment]: <> (```)

[comment]: <> (bash scripts/download_data.sh)

[comment]: <> (```)

[comment]: <> (This script should download and unpack the data automatically into the `data/synthetic_room_dataset` folder.  )

[comment]: <> (**Note**: We also provide **point-wise semantic labels** in the dataset, which might be useful.)


[comment]: <> (Alternatively, you can also preprocess the dataset yourself.)

[comment]: <> (To this end, you can:)

[comment]: <> (* download the ShapeNet dataset as described above.)

[comment]: <> (* check `scripts/dataset_synthetic_room/build_dataset.py`, modify the path and run the code.)

[comment]: <> (### Matterport3D)

[comment]: <> (Download Matterport3D dataset from [the official website]&#40;https://niessner.github.io/Matterport/&#41;. And then, use `scripts/dataset_matterport/build_dataset.py` to preprocess one of your favorite scenes. Put the processed data into `data/Matterport3D_processed` folder.)

[comment]: <> (### ScanNet)

[comment]: <> (Download ScanNet v2 data from the [official ScanNet website]&#40;https://github.com/ScanNet/ScanNet&#41;.)

[comment]: <> (Then, you can preprocess data with:)

[comment]: <> (`scripts/dataset_scannet/build_dataset.py` and put into `data/ScanNet` folder.  )

[comment]: <> (**Note**: Currently, the preprocess script normalizes ScanNet data to a unit cube for the comparison shown in the paper, but you can easily adapt the code to produce data with real-world metric. You can then use our fully convolutional model to run evaluation in a sliding-window manner.)

[comment]: <> (## Usage)

[comment]: <> (When you have installed all binary dependencies and obtained the preprocessed data, you are ready to run our pre-trained models and train new models from scratch.)

[comment]: <> (### Mesh Generation)

[comment]: <> (To generate meshes using a trained model, use)

[comment]: <> (```)

[comment]: <> (python generate.py CONFIG.yaml)

[comment]: <> (```)

[comment]: <> (where you replace `CONFIG.yaml` with the correct config file.)

[comment]: <> (**Use a pre-trained model**  )

[comment]: <> (The easiest way is to use a pre-trained model. You can do this by using one of the config files under the `pretrained` folders.)

[comment]: <> (For example, for 3D reconstruction from noisy point cloud with our 3-plane model on the synthetic room dataset, you can simply run:)

[comment]: <> (```)

[comment]: <> (python generate.py configs/pointcloud/pretrained/room_3plane.yaml)

[comment]: <> (```)

[comment]: <> (The script will automatically download the pretrained model and run the generation. You can find the outputs in the `out/.../generation_pretrained` folders)

[comment]: <> (Note that the config files are only for generation, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pretrained model.)


[comment]: <> (We provide the following pretrained models:)

[comment]: <> (```)

[comment]: <> (pointcloud/shapenet_1plane.pt)

[comment]: <> (pointcloud/shapenet_3plane.pt)

[comment]: <> (pointcloud/shapenet_grid32.pt)

[comment]: <> (pointcloud/shapenet_3plane_partial.pt)

[comment]: <> (pointcloud/shapenet_pointconv.pt)

[comment]: <> (pointcloud/room_1plane.pt)

[comment]: <> (pointcloud/room_3plane.pt)

[comment]: <> (pointcloud/room_grid32.pt)

[comment]: <> (pointcloud/room_grid64.pt)

[comment]: <> (pointcloud/room_combine.pt)

[comment]: <> (pointcloud/room_pointconv.pt)

[comment]: <> (pointcloud_crop/room_grid64.pt)

[comment]: <> (voxel/voxel_shapenet_1plane.pt)

[comment]: <> (voxel/voxel_shapenet_3plane.pt)

[comment]: <> (voxel/voxel_shapenet_grid32.pt)

[comment]: <> (```)

[comment]: <> (### Evaluation)

[comment]: <> (For evaluation of the models, we provide the script `eval_meshes.py`. You can run it using:)

[comment]: <> (```)

[comment]: <> (python eval_meshes.py CONFIG.yaml)

[comment]: <> (```)

[comment]: <> (The script takes the meshes generated in the previous step and evaluates them using a standardized protocol. The output will be written to `.pkl/.csv` files in the corresponding generation folder which can be processed using [pandas]&#40;https://pandas.pydata.org/&#41;.)

[comment]: <> (### Training)

[comment]: <> (Finally, to train a new network from scratch, run:)

[comment]: <> (```)

[comment]: <> (python train.py CONFIG.yaml)

[comment]: <> (```)

[comment]: <> (For available training options, please take a look at `configs/default.yaml`.)

[comment]: <> (## Further Information)

[comment]: <> (Please also check out the following concurrent works that either tackle similar problems or share similar ideas:)

[comment]: <> (- [[CVPR 2020] Jiang et al. - Local Implicit Grid Representations for 3D Scenes]&#40;https://arxiv.org/abs/2003.08981&#41;)

[comment]: <> (- [[CVPR 2020] Chibane et al. Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion]&#40;https://arxiv.org/abs/2003.01456&#41;)

[comment]: <> (- [[ECCV 2020] Chabra et al. - Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction]&#40;https://arxiv.org/abs/2003.10983&#41;)

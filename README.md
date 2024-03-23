# ARO Net

## Download

```bash
https://1drv.ms/u/s!AkDQSKsmQQCghcQfKvYspGIW031LeA?e=NDzEi6
->
data.zip
```

The layout of ``data`` dir is: 

```
ARO-Net
├── data
│   ├── abc
    │   │   │──00_meshes
    │   │   │──01_pcds
    │   │   │──02_qry_pts
    │   │   │──03_qry_dists
    │   │   │──04_splits
    ├── shapenet
    │   │   │──00_meshes
    │   │   │   │──02691156
    │   │   │   │──03001627
    │   │   │──01_pcds
    │   │   │──02_qry_pts
    │   │   │──03_qry_dists
    │   │   │──04_splits
    ├── anchors
```

### Customize Your Own Dataset

We followed the [script](https://github.com/ErlerPhilipp/points2surf/blob/2af6e0facf58422ed12e0c676c70199cd0dfbb43/make_dataset.py#L406C24-L406C46) in [Points2Surf](https://github.com/ErlerPhilipp/points2surf) to sample `qry_pts` and calculate their `signed distance`. Note that we did not use their way of normalization, our way of normalization can be found in this [link](https://github.com/yizhiwang96/ARO-Net/blob/main/data_processing/norm_and_sample_mesh.py), where the length of body diagonal of 3D bounding box is normalized to 1.

## Setup

```bash
conda create -n aro python=3.10
conda activate aro
./setup.sh
```

## Run

```bash
python demo.py
```

```
# ShapeNet Airplane (trained w/ IM-Net data)
python reconstruct.py --name_exp pretrained_chairs --name_ckpt aronet_chairs_gt_imnet.ckpt --name_dataset shapenet --categories_test 02691156, --use_dist_hit --n_pts_test 2048 --mc_threshold 0.5

# ShapeNet Airplane (trained w/ OCC-Net data)
python reconstruct.py --name_exp pretrained_chairs --name_ckpt aronet_chairs_gt_occnet.ckpt --name_dataset shapenet --categories_test 02691156, --norm_coord --n_pts_test 2048 --mc_threshold 0.2
```

## Train

We use Fibonacci sampling to generate 48 anchors for our ARO-Net. Other anchor settings can generated with `gen_anc.py`.

To train ARO-Net on ABC dataset or ShapeNet:
```
python cal_hit_dist.py
python train.py --name_exp base_model_chairs --name_dataset shapenet --categories_train 03001627, --use_dist_hit --norm_coord --gt_source occnet
```
To train ARO-Net on single shape with data augmentation:
```
python train.py --name_exp base_model --name_dataset single --name_single fertility
```

Check all training options in `options.py`. You need one NVIDIA A100 (80G) to train ARO-Net under the configurations in `options.py`. You can set the `n_bs` and `n_qry` to fit to your GPU capacity. set `n_bs` to `4` and `n_qry` to `256` will cost ~20GB video memory.

## Evaluation

To reconstruct meshes on test sets:
```
# ShapeNet Airplane
python reconstruct.py --name_exp base_model_chairs --name_ckpt 600_301101_xxx_xxx.ckpt --name_dataset shapenet --name_dataset 02691156, --norm_coord
```

To evalute HD, CD, and IoU:
```
# ShapeNet
python eval_metrics.py --name_exp base_model_chairs --name_dataset shapenet --categories_test 03001627,
python eval_metrics.py --name_exp base_model_chairs --name_dataset shapenet --categories_test 02691156,
```

## Enjoy it~

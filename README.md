# On Exploring PDE Modeling for Point Cloud Video Representation Learning

## Action Recognition results on MSR-Action3D
| Methods                                | 4     | 8     | 12    | 16    | 20    | 24   |        
|----------------------------------------|-------|-------|-------|-------|-------|------|
| MeteorNet                              | 78.11 | 81.14 | 86.53 | 88.21 | -     | 88.50 |
| P4Transformer                          | 80.13 | 83.17 | 87.54 | 89.56 | 90.24 | 90.94 |
| PSTNet                                 | 81.14 | 83.50 | 87.88 | 89.90 | -     | 91.20 |
| SequentialPointNet                     | 77.66 | 86.45 | 88.64 | 89.56 | 91.21 | 91.94 |
| PSTNet++                               | 81.53 | 83.50 | 88.15 | 90.24 | -     | 92.68 |
| Anchor-Based Spatio-Temporal Attention | 80.13 | 87.54 | 89.90 | 91.24 | -     | 93.03 |
| Kinet                                  | 79.80 | 83.84 | 88.53 | 91.92 | -     | 93.27 |
| PST-Transformer                        | 81.14 | 83.97 | 88.15 | 91.98 | -     | 93.73 |
| Motion PointNet (Ours)                 | 79.46 | 85.88 | 90.57 | 93.33 | -     | 97.52 |
    

## Installation

```
conda create -n motion python=3.7
conda activate motion
```

Install
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```


Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
cd cpp/pointnet2_batch
python setup.py install
```

## Download the dataset
Down the MSRAction-3D dataset from [here](https://drive.google.com/file/d/1djwAK3oZTAIFbCz531eClxINmsZgGO_H/view?usp=sharing). Unzip the file and get the ```Depth``` folder. Then run the following commend:
```
python msr_preprocess.py --input_dir /path/to/Depth --output_dir /path/to/data
```

## Training on MSRAction-3D from the start
modify the ```resume``` in [config](./configs) to ```None``` and run
```
torchrun --nproc_per_node=2 --master_port 33333 train.py --cfg configs/msr_motion_finetune.py
```

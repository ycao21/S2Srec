# S2Srec
This repository contains the code to train and evaluate models from the paper:  
S2SRec: Set-to-Set Recommendation for Recipe Completion using Set Transformer and Multi-Task Learning
# Requirements
```shell
pip install -r requirements.txt
```
# Data
## Data preparation
To prepare training data from scratch, run:
```shell
python data_preparation.py
```
Please unzip `data/map_data.zip` and put files under `data/map`

# Model
To train the model:
```shell
bash run.sh
```
The model will be saved in `--snapshots` path.

# Credit
The backbone of this framework is based on [torralba-lab/im2recipe-Pytorch](https://github.com/torralba-lab/im2recipe-Pytorch)

The implementation of Set Transformer is based on [TropComplique/set-transformer](https://github.com/TropComplique/set-transformer).

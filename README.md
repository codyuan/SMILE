# Good Better Best: Self-Motivated Imitation Learning for noisy Demonstrations

All relevant citations for methods used are found in the paper's list of references（https://arxiv.org/abs/2310.15815 ）

## Requirements

We recommend using pacakge manager [pip](https://pip.pypa.io/en/stable/) as well as 
[cuda](https://developer.nvidia.com/cuda-toolkit) to install the relative packages:

All packages required is provided in **smile.yaml**

## Installation

First create a new environment that has python and pytorch. 
We provide corresponding command to create an enviornment called smile using **smile.yaml**:

```bash
conda env create -f smile.yaml
conda activate smile
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/.mujoco/mujoco200/bin
```

Please make sure that [MuJoCo](https://github.com/deepmind/mujoco) and [mujoco-py](https://github.com/openai/mujoco-py) is installed finely. You can click the link to check the instructions of install.


## Usage
Before starting training of SMILE, one must first load an expert policy to generate the diverse-quality demonstrations. Expert policies are pretrained by SAC implemented in [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Codes to load their trained parameters and generate demonstrations is provided in **scripts/SAC_collect.py**.

Therefore, to collect diverse-quality demonstrations on Hopper, one can easily run:

```bash
cd scripts
python SAC_collect.py --env_name Hopper-v2
```

Then, the SMILE can be trained by collected demonstrations. All codes of SMILE algorithm is placed in **/model**. Besides, some useful functions and classes is uniformly placed in **/util**.

To train SMILE, one can run the code provided in **/scripts/train.py** to replicate MuJoCo experiments. Plus, one can achieve some variants of SMILE through setting different hyper-parameters.

For example, to train whole framework of SMILE on Hopper, one can run:
```bash
cd scripts
python train.py --env_name Hopper-v2 --denoiser_loss_type l1 --policy_loss_type l2  
```

To train SMILE w/o filter, one can run:
```bash
cd scripts
python train.py --env_name Hopper-v2 --denoiser_loss_type l1 --policy_loss_type l2 --no_filtering True 
```

To train SMILE with naive reverse generation, one can run:
```bash
cd scripts
python train.py --env_name Hopper-v2 --denoiser_loss_type l1 --policy_loss_type l2 --naive_reverse True
```


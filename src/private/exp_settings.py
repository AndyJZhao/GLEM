from pathlib import Path

# ! Servers
LINUX_HOME = str(Path.home())
CONDA_ENV_NAME = 'ct'
CONDA_PATH = f'{LINUX_HOME}/miniconda/envs/{CONDA_ENV_NAME}'
NV_HTOP_FILE = f"{CONDA_PATH}/bin/nvidia-htop.py"
SV_INIT_CMDS = [
    f'source {LINUX_HOME}/miniconda/etc/profile.d/conda.sh;conda activate {CONDA_ENV_NAME}',
    f'alias tr_lm="python src/models/LMs/trainLM.py"',
]

# ! Git settings
GIT_ACCOUNT = 'AndyJZhao'
GIT_TOKEN = 'GHSAT0AAAAAABRRWTPVFBISNLHINRXUMIWKYSKKV3Q'

# ! Wandb settings
# WANDB_API_KEY = 'e9e1d0ee12e86ea7a6f8b3fb106dc97dc8f22604'
# WANDB_DIR = 'wandb'
# WANDB_ENTITY = 'jianan'
# WANDB_PROJ = 'CT-Debug'
WANDB_API_KEY = '6d7419c4c0209ac98b0ae8f24554c9460a5ae0a9'
WANDB_DIR = 'wandb_temp'
WANDB_ENTITY = 'cirtraining'
WANDB_PROJ = 'CT-Debug'

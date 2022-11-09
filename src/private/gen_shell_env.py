import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from utils.settings import *
from private.exp_settings import *

env_vars = {
    # PATHS
    'LD_LIBRARY_PATH': f'{CONDA_PATH}/lib/python3.8/site-packages/torch/lib/',
    'LP': PROJ_DIR,  # Local Path
    'TEMP_DIR': TEMP_DIR,
    'MNT_DIR': MNT_DIR,
    'PROJ_NAME': PROJ_NAME,  # PROJ Name
    'WANDB_DIR': WANDB_DIR,
    'WANDB_ENTITY': WANDB_ENTITY,
    'WANDB_API_KEY': WANDB_API_KEY,
    'HTOP_FILE': NV_HTOP_FILE,  # Nvidia-htop file
    'TR': f'{PROJ_DIR}src/models/GLEM/trainCT.py'
}

server_setting_file = f'{PROJ_DIR}/shell_env.sh'
with open(server_setting_file, 'w') as f:
    for var_name, var_val in env_vars.items():
        f.write(f'export {var_name}="{var_val}"\n')

    for cmd in SV_INIT_CMDS:
        f.write(f'{cmd}\n')
    print()

import os
import sys
import isaacgym

from afforddp.workspace.base_workspace import BaseWorkspace
from afforddp.utils.seed import set_np_formatting, set_seed
import pathlib
from omegaconf import OmegaConf
import hydra
from termcolor import cprint
import numpy as np
import random
import torch

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

# allow for detecting segmentation fault
# import faulthandler
# faulthandler.enable()
# cprint("[fault handler enabled]", "cyan")

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'afforddp', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.eval()


if __name__ == "__main__":

    set_np_formatting()
    set_seed(seed=1)
    main()






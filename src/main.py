from __future__ import annotations

import os
import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from devnet.config import TrainerConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)


@hydra.main(config_path="conf", config_name="main")
def main(config: TrainerConfig):
    dataroot = config.dataroot
    if not os.path.isabs(dataroot):
        config.dataroot = os.path.join(hydra.utils.get_original_cwd(), dataroot)

    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
